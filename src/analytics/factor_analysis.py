"""
AlphaForge — ETF-Proxy Factor Analysis
=======================================
Simplified educational-grade factor exposure analysis using liquid
U.S. ETFs as factor proxies.

This is NOT a formal Fama-French model.  It uses publicly available
ETF returns as approximate proxies for systematic risk factors and is
intended as a research-only tool.

Factor proxies
--------------
Market  (MKT):  SPY  — broad U.S. equity market
Small Cap:      IWM  — Russell 2000
Growth:         QQQ  — NASDAQ-100 / growth tilt
Value:          VTV  — Vanguard Value
Bonds:          AGG  — U.S. Aggregate Bond
Gold:           GLD  — SPDR Gold Shares
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# ── Factor proxy definitions ──────────────────────────────────────────────────

FACTOR_PROXIES: dict[str, str] = {
    "Market (SPY)":    "SPY",
    "Small Cap (IWM)": "IWM",
    "Growth (QQQ)":    "QQQ",
    "Value (VTV)":     "VTV",
    "Bonds (AGG)":     "AGG",
    "Gold (GLD)":      "GLD",
}

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "Market (SPY)":    "Broad U.S. equity market beta",
    "Small Cap (IWM)": "Small-cap size premium proxy",
    "Growth (QQQ)":    "Growth/momentum tilt (NASDAQ-100)",
    "Value (VTV)":     "Value factor proxy",
    "Bonds (AGG)":     "Fixed-income / duration exposure",
    "Gold (GLD)":      "Safe-haven / inflation hedge",
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_factor_returns(
    start: str,
    end: str,
    factors: list[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Download daily total returns for factor proxy ETFs via yfinance.

    Parameters
    ----------
    start, end : str
        Date range in "YYYY-MM-DD" format.
    factors : list[str] | None
        Subset of FACTOR_PROXIES keys to fetch.
        ``None`` fetches all six proxies.

    Returns
    -------
    dict[str, pd.Series]
        factor_label → daily return series.  Missing proxies are omitted.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is required for factor analysis.")
        return {}

    selected = factors if factors else list(FACTOR_PROXIES.keys())
    tickers  = [FACTOR_PROXIES[f] for f in selected if f in FACTOR_PROXIES]
    labels   = [f for f in selected if f in FACTOR_PROXIES]

    if not tickers:
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False,
            )

        # Handle single vs multi-ticker download
        if len(tickers) == 1:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close = raw["Close"]

        result: dict[str, pd.Series] = {}
        for label, ticker in zip(labels, tickers):
            if ticker in close.columns:
                rets = close[ticker].dropna().pct_change().dropna()
                result[label] = rets

        logger.info("Fetched %d factor proxies.", len(result))
        return result

    except Exception as exc:
        logger.error("Factor proxy download failed: %s", exc)
        return {}


# ── OLS regression ────────────────────────────────────────────────────────────

def run_factor_regression(
    strategy_returns: pd.Series,
    factor_returns: dict[str, pd.Series],
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Regress strategy excess returns against factor proxy returns (OLS).

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    factor_returns : dict[str, pd.Series]
        factor_label → daily return series.
    risk_free_rate : float
        Annual risk-free rate for excess return calculation.

    Returns
    -------
    dict with keys
        alpha, alpha_ann, factor_loadings, t_stats, p_values,
        r_squared, adj_r_squared, n_obs, factors_used.
        Empty dict if there is insufficient data.
    """
    if not factor_returns:
        return {}

    # Align on common dates
    data = pd.DataFrame({"strategy": strategy_returns})
    for name, series in factor_returns.items():
        data[name] = series
    data = data.dropna()

    if len(data) < 30:
        logger.warning("Factor regression needs ≥30 observations, got %d.", len(data))
        return {}

    daily_rf   = risk_free_rate / 252.0
    y          = (data["strategy"] - daily_rf).values
    fac_names  = [c for c in data.columns if c != "strategy"]
    X_raw      = data[fac_names].values
    n, k       = len(y), len(fac_names)

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), X_raw])

    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    y_hat  = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)

    mse   = ss_res / max(n - k - 1, 1)
    cov_b = mse * np.linalg.pinv(X.T @ X)
    se    = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    t_vec = np.where(se > 0, coeffs / se, np.nan)
    p_vec = 2.0 * (1.0 - t_dist.cdf(np.abs(t_vec), df=max(n - k - 1, 1)))

    all_names = ["alpha"] + fac_names

    return {
        "alpha":           float(coeffs[0]),
        "alpha_ann":       float(coeffs[0] * 252),
        "factor_loadings": dict(zip(fac_names, coeffs[1:].tolist())),
        "t_stats":         dict(zip(all_names, t_vec.tolist())),
        "p_values":        dict(zip(all_names, p_vec.tolist())),
        "r_squared":       float(r2),
        "adj_r_squared":   float(adj_r2),
        "n_obs":           n,
        "factors_used":    fac_names,
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def full_factor_analysis(
    strategy_returns: pd.Series,
    start: str,
    end: str,
    factors: list[str] | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Fetch factor proxies and run the regression in one call.

    Returns the regression dict, or {} on failure.
    """
    proxy_returns = fetch_factor_returns(start, end, factors)
    if not proxy_returns:
        return {}
    return run_factor_regression(strategy_returns, proxy_returns, risk_free_rate)
