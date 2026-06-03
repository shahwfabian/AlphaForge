"""
Portfolio risk decomposition analytics.

Modules
-------
rolling_beta_attribution   — decompose portfolio variance into systematic + idiosyncratic
volatility_attribution     — rolling contribution of market vs residual vol
sector_exposure            — sector weight map for a portfolio of tickers
beta_decomposition_series  — time-series of rolling beta and alpha
factor_sensitivity         — OLS regression of returns on constructed risk factors
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ── Rolling beta / alpha ──────────────────────────────────────────────────────

def rolling_beta_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling OLS beta and alpha over a sliding window.

    Beta   = Cov(r_s, r_b) / Var(r_b)  — market sensitivity
    Alpha  = mean(r_s) - beta * mean(r_b)  — daily excess return

    Parameters
    ----------
    strategy_returns : pd.Series
    benchmark_returns : pd.Series
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: Beta, Alpha (daily), Alpha_Ann, R_Squared.
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1, join="inner",
    ).dropna()

    betas, alphas, r2s = [], [], []

    for i in range(len(aligned)):
        if i < window - 1:
            betas.append(np.nan)
            alphas.append(np.nan)
            r2s.append(np.nan)
            continue

        window_s = aligned["s"].iloc[i - window + 1: i + 1]
        window_b = aligned["b"].iloc[i - window + 1: i + 1]

        slope, intercept, r_val, _, _ = stats.linregress(window_b, window_s)
        betas.append(slope)
        alphas.append(intercept)
        r2s.append(r_val ** 2)

    result = pd.DataFrame(
        {"Beta": betas, "Alpha": alphas, "R_Squared": r2s},
        index=aligned.index,
    )
    result["Alpha_Ann"] = result["Alpha"] * TRADING_DAYS_PER_YEAR
    return result


# ── Variance decomposition ────────────────────────────────────────────────────

def variance_decomposition(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Decompose portfolio variance into systematic and idiosyncratic components.

    Total Variance   = Systematic Variance + Idiosyncratic Variance
    Systematic Var   = β² × Var(benchmark)
    Idiosyncratic    = Var(strategy) - Systematic Var

    Parameters
    ----------
    strategy_returns : pd.Series
    benchmark_returns : pd.Series
    window : int

    Returns
    -------
    pd.DataFrame
        Columns: Total_Variance, Systematic_Variance, Idiosyncratic_Variance,
                 Systematic_Pct, Idiosyncratic_Pct, Beta.
    """
    roll_ba = rolling_beta_alpha(strategy_returns, benchmark_returns, window)

    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b"),
         roll_ba["Beta"]],
        axis=1, join="inner",
    ).dropna(subset=["Beta"])

    total_var  = aligned["s"].rolling(window).var()
    bm_var     = aligned["b"].rolling(window).var()
    systematic = (aligned["Beta"] ** 2) * bm_var
    idiosync   = (total_var - systematic).clip(lower=0)

    result = pd.DataFrame({
        "Total_Variance":          total_var,
        "Systematic_Variance":     systematic,
        "Idiosyncratic_Variance":  idiosync,
        "Beta":                    aligned["Beta"],
    }, index=aligned.index)

    result["Systematic_Pct"] = (
        result["Systematic_Variance"] / result["Total_Variance"]
    ).clip(0, 1)
    result["Idiosyncratic_Pct"] = 1 - result["Systematic_Pct"]

    return result


# ── Rolling volatility attribution ────────────────────────────────────────────

def rolling_volatility_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Decompose rolling annualised volatility into market and residual components.

    Returns
    -------
    pd.DataFrame
        Columns: Strategy_Vol, Benchmark_Vol, Market_Vol_Contribution,
                 Residual_Vol_Contribution, Beta.
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1, join="inner",
    ).dropna()

    scale = np.sqrt(TRADING_DAYS_PER_YEAR)

    s_vol = aligned["s"].rolling(window).std() * scale
    b_vol = aligned["b"].rolling(window).std() * scale

    # Rolling beta (faster approximation via rolling covariance / variance)
    rolling_cov = aligned["s"].rolling(window).cov(aligned["b"])
    rolling_var = aligned["b"].rolling(window).var()
    rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)

    # Market contribution to strategy vol = beta × benchmark_vol
    market_contrib = rolling_beta.abs() * b_vol

    # Residual = strategy_vol - market_contrib (floored at 0)
    residual_contrib = (s_vol - market_contrib).clip(lower=0)

    return pd.DataFrame({
        "Strategy_Vol":              s_vol,
        "Benchmark_Vol":             b_vol,
        "Market_Vol_Contribution":   market_contrib,
        "Residual_Vol_Contribution": residual_contrib,
        "Beta":                      rolling_beta,
    }, index=aligned.index)


# ── Factor sensitivity (multi-factor regression) ─────────────────────────────

def factor_sensitivity(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    lookback_days: int = 90,
) -> dict:
    """
    Regress strategy excess returns on three constructed risk factors.

    Factors (all computable from existing data — no extra downloads):
    ──────────────────────────────────────────────────────────────────
    MKT  : benchmark excess return  (E[r_b] - rf)
    MOM  : trailing lookback_days return of the benchmark (momentum factor proxy)
    VOL  : change in 21-day rolling volatility of the benchmark (vol-of-vol)

    Outputs
    -------
    dict with keys: factor_loadings, t_stats, p_values, r_squared,
                    alpha, alpha_t_stat, factor_names.
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1, join="inner",
    ).dropna()

    if len(aligned) < lookback_days + 30:
        logger.warning("Insufficient data for factor regression (%d rows).", len(aligned))
        return {}

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # Factor 1: Market excess return
    mkt = aligned["b"] - daily_rf

    # Factor 2: Trailing momentum (trailing lookback return, lagged 1 day)
    mom = aligned["b"].rolling(lookback_days).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    ).shift(1)

    # Factor 3: Volatility change (21-day vol change, lagged 1 day)
    roll_vol = aligned["b"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol_chg  = roll_vol.diff().shift(1)

    # Strategy excess return (dependent variable)
    y = aligned["s"] - daily_rf

    # Build factor matrix and drop NaN rows
    factors = pd.concat(
        [y.rename("y"), mkt.rename("MKT"), mom.rename("MOM"), vol_chg.rename("VOL")],
        axis=1,
    ).dropna()

    if len(factors) < 30:
        return {}

    Y = factors["y"].values
    X = np.column_stack([
        np.ones(len(factors)),
        factors["MKT"].values,
        factors["MOM"].values,
        factors["VOL"].values,
    ])

    # OLS: β = (XᵀX)⁻¹ Xᵀy
    try:
        betas, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    # Standard errors and t-statistics
    n, k = len(Y), X.shape[1]
    y_hat = X @ betas
    resid = Y - y_hat
    s2 = np.dot(resid, resid) / (n - k)
    cov_b = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_b))
    t_stats = betas / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    ss_res = np.dot(resid, resid)
    ss_tot = np.dot(Y - Y.mean(), Y - Y.mean())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    factor_names = ["MKT", "MOM", "VOL"]
    return {
        "alpha":            round(float(betas[0]) * TRADING_DAYS_PER_YEAR, 6),
        "alpha_t_stat":     round(float(t_stats[0]), 4),
        "alpha_p_value":    round(float(p_values[0]), 4),
        "factor_loadings":  dict(zip(factor_names, [round(float(b), 4) for b in betas[1:]])),
        "t_stats":          dict(zip(factor_names, [round(float(t), 4) for t in t_stats[1:]])),
        "p_values":         dict(zip(factor_names, [round(float(p), 4) for p in p_values[1:]])),
        "r_squared":        round(float(r_squared), 4),
        "n_observations":   int(n),
        "factor_names":     factor_names,
    }


# ── Sector exposure ───────────────────────────────────────────────────────────

# Static sector mapping for common ETFs and large-cap stocks.
# Avoids live yfinance .info() calls (slow, unreliable).
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "V": "Financials", "MA": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # ETFs
    "SPY": "Broad Market", "QQQ": "Technology", "IWM": "Broad Market",
    "DIA": "Broad Market", "XLK": "Technology", "XLF": "Financials",
    "XLE": "Energy", "XLV": "Healthcare", "XLI": "Industrials",
    "GLD": "Commodities", "TLT": "Fixed Income", "AGG": "Fixed Income",
    "BTC-USD": "Crypto", "ETH-USD": "Crypto",
}


def sector_exposure(
    tickers: list[str],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute portfolio-level sector exposure.

    Parameters
    ----------
    tickers : list[str]
        Portfolio constituent tickers.
    weights : list[float] or None
        Portfolio weights (sum to 1). Equal-weight if None.

    Returns
    -------
    pd.DataFrame
        Columns: Ticker, Sector, Weight, Sector_Weight.
    """
    if not tickers:
        return pd.DataFrame()

    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)

    rows = []
    for ticker, w in zip(tickers, weights):
        sector = SECTOR_MAP.get(ticker.upper(), "Unknown")
        rows.append({"Ticker": ticker, "Sector": sector, "Weight": w})

    df = pd.DataFrame(rows)
    sector_agg = df.groupby("Sector")["Weight"].sum().reset_index()
    sector_agg.rename(columns={"Weight": "Sector_Weight"}, inplace=True)
    sector_agg.sort_values("Sector_Weight", ascending=False, inplace=True)

    return sector_agg


def full_risk_decomposition(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    ticker: str,
    window: int = 63,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Run all risk decomposition analyses and return a unified results dict.

    Returns
    -------
    dict with keys:
        rolling_beta_alpha, variance_decomp, vol_attribution,
        factor_model, sector_exposure.
    """
    rba   = rolling_beta_alpha(strategy_returns, benchmark_returns, window)
    vd    = variance_decomposition(strategy_returns, benchmark_returns, window)
    va    = rolling_volatility_attribution(strategy_returns, benchmark_returns)
    fm    = factor_sensitivity(strategy_returns, benchmark_returns, risk_free_rate)
    se    = sector_exposure([ticker])

    return {
        "rolling_beta_alpha":  rba,
        "variance_decomp":     vd,
        "vol_attribution":     va,
        "factor_model":        fm,
        "sector_exposure":     se,
    }
