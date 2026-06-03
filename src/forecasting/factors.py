"""
AlphaForge — Factor-Based Scenario Forecasting
================================================
Estimates conditional future scenarios for a target asset as a function of
factor movements, using OLS factor regression combined with Monte Carlo
simulation.

Factors (ETF proxies)
---------------------
  SPY  — Broad market (S&P 500)
  QQQ  — Technology / Growth
  IWM  — Small-cap
  VTV  — Value
  TLT  — Long-duration bonds
  GLD  — Gold / inflation hedge

Methodology
-----------
1. Regress excess asset returns on factor excess returns:
     r_asset = α + β₁·r_SPY + β₂·r_QQQ + … + ε

2. Generate correlated factor return scenarios using a multivariate
   Normal calibrated to the factor covariance matrix.

3. Propagate factor scenarios through the regression equation to get
   asset return scenarios.  This gives a *conditional* distribution:
     P(r_asset | r_factors = scenario)

4. Build the factor contribution decomposition:
   how much of the asset's return variability is attributable to each factor.

Output
------
Conditional scenario distributions and factor sensitivity report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Factor scenarios are conditional on the regression model and the "
    "assumption that historical factor relationships persist. "
    "Factor betas are estimated from historical data and may not reflect "
    "current market conditions.  This is not investment advice."
)

FACTOR_PROXIES: dict[str, str] = {
    "Market (SPY)":   "SPY",
    "Tech/Growth (QQQ)": "QQQ",
    "Small Cap (IWM)": "IWM",
    "Value (VTV)":    "VTV",
    "Bonds (TLT)":    "TLT",
    "Gold (GLD)":     "GLD",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FactorForecastResult:
    ticker: str
    factors_used: list[str]

    # Regression outputs
    alpha_ann: float                   # annualised Jensen's α
    betas: dict[str, float]            # factor → beta
    r_squared: float
    adj_r_squared: float
    t_stats: dict[str, float]
    p_values: dict[str, float]
    residual_vol_ann: float            # idiosyncratic volatility

    # Factor contribution
    contribution_pct: dict[str, float] # % of total variance from each factor

    # Conditional scenarios
    scenarios_df: pd.DataFrame         # factor shock → expected asset return + CI

    # Simulated conditional paths
    scenario_paths: dict[str, np.ndarray]  # scenario_name → (n_paths,) terminal rets

    disclaimer: str = field(default=_DISCLAIMER, repr=False)

    def top_factors(self, n: int = 3) -> list[str]:
        return sorted(self.betas, key=lambda k: abs(self.betas[k]), reverse=True)[:n]

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for f in self.betas:
            rows.append({
                "Factor":        f,
                "Beta":          f"{self.betas[f]:+.3f}",
                "t-stat":        f"{self.t_stats[f]:+.2f}",
                "p-value":       f"{self.p_values[f]:.4f}",
                "Contribution":  f"{self.contribution_pct.get(f, 0.0)*100:.1f}%",
                "Significant":   "Yes" if self.p_values[f] < 0.05 else "No",
            })
        return pd.DataFrame(rows)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_factor_returns(
    start: str,
    end: str,
    factors: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Fetch factor ETF returns via yfinance."""
    import yfinance as yf
    tickers = list(factors or FACTOR_PROXIES.values())
    result: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            px = yf.download(t, start=start, end=end, progress=False,
                             auto_adjust=True)["Close"].squeeze()
            if not px.empty:
                result[t] = np.log(px / px.shift(1)).dropna()
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", t, exc)
    return result


# ── OLS factor regression ─────────────────────────────────────────────────────

def _run_regression(
    asset_rets: pd.Series,
    factor_rets: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """OLS regression of asset excess returns on factor excess returns."""
    rf_daily = risk_free_rate / 252.0
    y = (asset_rets - rf_daily).dropna()
    X = (factor_rets - rf_daily).reindex(y.index).dropna()
    y = y.reindex(X.index)

    if len(y) < 30:
        raise ValueError("Insufficient overlapping data for factor regression.")

    X_mat = np.column_stack([np.ones(len(X)), X.values])
    b, res, rank, sv = np.linalg.lstsq(X_mat, y.values, rcond=None)

    y_hat   = X_mat @ b
    resid   = y.values - y_hat
    n, k    = len(y), X_mat.shape[1]
    sse     = float(np.sum(resid ** 2))
    sst     = float(np.sum((y.values - y.values.mean()) ** 2))
    r2      = 1.0 - sse / sst if sst > 0 else 0.0
    r2_adj  = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0
    sigma2  = sse / (n - k) if n > k else sse
    cov_b   = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)
    se      = np.sqrt(np.diag(cov_b))
    t_stats = b / (se + 1e-12)
    p_vals  = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_stats]

    return dict(
        alpha_daily = float(b[0]),
        betas       = dict(zip(X.columns, b[1:].tolist())),
        t_stats     = dict(zip(["alpha"] + list(X.columns), t_stats.tolist())),
        p_values    = dict(zip(["alpha"] + list(X.columns), p_vals)),
        r_squared   = float(r2),
        adj_r_squared = float(r2_adj),
        residuals   = pd.Series(resid, index=y.index),
        factors     = X,
    )


# ── Variance decomposition ────────────────────────────────────────────────────

def _variance_decomposition(
    betas: dict[str, float],
    factor_cov: np.ndarray,
    factor_names: list[str],
    residual_var: float,
) -> dict[str, float]:
    """
    Decompose total return variance into factor contributions.
    Var(r) ≈ β^T Σ_F β + σ²_ε
    """
    b = np.array([betas[f] for f in factor_names])
    total_factor_var = float(b @ factor_cov @ b)
    total_var        = total_factor_var + residual_var
    if total_var < 1e-12:
        return {f: 0.0 for f in factor_names}
    # Marginal contribution: (Σ_F β)_i * β_i
    fac_contrib = factor_cov @ b * b
    result = {
        f: float(fac_contrib[i] / total_var)
        for i, f in enumerate(factor_names)
    }
    return result


# ── Conditional scenario simulation ──────────────────────────────────────────

_SHOCK_SCENARIOS: dict[str, dict[str, float]] = {
    "Bull Run (+20% market)":     {"SPY": +0.20},
    "Mild Rally (+10% market)":   {"SPY": +0.10},
    "Base Case (0%)":             {"SPY":  0.00},
    "Mild Correction (-10%)":     {"SPY": -0.10},
    "Bear Market (-20%)":         {"SPY": -0.20},
    "Tech Selloff (-20% QQQ)":    {"QQQ": -0.20},
    "Rate Spike (TLT -15%)":      {"TLT": -0.15},
    "Gold Rally (+15% GLD)":      {"GLD": +0.15},
}


# ── Public API ────────────────────────────────────────────────────────────────

def run_factor_forecast(
    asset_prices: pd.Series,
    start: str | None = None,
    end:   str | None = None,
    horizon: int = 63,
    n_paths: int = 1_000,
    risk_free_rate: float = 0.0,
    factors: list[str] | None = None,
    seed: int = 42,
) -> FactorForecastResult:
    """
    Fit factor model and generate conditional scenario forecasts.

    Parameters
    ----------
    asset_prices     : historical close prices
    start / end      : date range for factor data fetch
    horizon          : forecast horizon in trading days
    n_paths          : conditional simulation paths per scenario
    risk_free_rate   : annualised risk-free rate
    factors          : list of factor ETF tickers (default: all 6)
    seed             : RNG seed

    Returns
    -------
    FactorForecastResult
    """
    rng     = np.random.default_rng(seed)
    ticker  = str(asset_prices.name or "ASSET")
    fac_tickers = list(factors or FACTOR_PROXIES.values())

    asset_idx = asset_prices.index
    if start is None:
        start = str(asset_idx[0].date() if hasattr(asset_idx[0], "date") else asset_idx[0])
    if end is None:
        end   = str(asset_idx[-1].date() if hasattr(asset_idx[-1], "date") else asset_idx[-1])

    asset_rets = np.log(asset_prices / asset_prices.shift(1)).dropna()
    asset_rets.name = ticker

    # ── Fetch factors ──────────────────────────────────────────────────────────
    raw_factors = _fetch_factor_returns(start, end, fac_tickers)
    if not raw_factors:
        raise ValueError("No factor data could be fetched.")

    factor_df = pd.DataFrame(raw_factors).dropna()
    available = [t for t in fac_tickers if t in factor_df.columns]
    factor_df = factor_df[available]

    if len(available) < 2:
        raise ValueError("Need at least 2 factors for regression.")

    # ── OLS regression ─────────────────────────────────────────────────────────
    reg = _run_regression(asset_rets, factor_df, risk_free_rate)
    betas     = reg["betas"]
    alpha_ann = reg["alpha_daily"] * 252.0
    resid_vol_ann = float(reg["residuals"].std() * np.sqrt(252))

    factor_cov = factor_df.cov().values * 252  # annualised covariance

    contrib = _variance_decomposition(
        betas, factor_cov, available, (resid_vol_ann / np.sqrt(252)) ** 2 * 252
    )

    # ── Conditional scenario table ─────────────────────────────────────────────
    T_years = horizon / 252.0
    rf_period = risk_free_rate * T_years

    scen_rows: list[dict] = []
    scen_paths: dict[str, np.ndarray] = {}

    for scen_name, shocks in _SHOCK_SCENARIOS.items():
        # Expected factor returns given scenario shocks
        fac_ret_expected = {}
        for f in available:
            hist_mean = float(factor_df[f].mean()) * horizon
            fac_ret_expected[f] = shocks.get(f, hist_mean)

        # Expected asset return
        asset_ret_exp = reg["alpha_daily"] * horizon + sum(
            betas.get(f, 0.0) * fac_ret_expected[f]
            for f in available
        )

        # Simulate conditional paths: add residual noise
        resid_std = float(reg["residuals"].std()) * np.sqrt(horizon)
        paths     = asset_ret_exp + rng.normal(0, resid_std, n_paths)

        # Terminal price distribution
        S0        = float(asset_prices.iloc[-1])
        terminal  = S0 * np.exp(paths)

        pct05 = float(np.percentile(terminal, 5))
        pct95 = float(np.percentile(terminal, 95))
        median = float(np.median(terminal))

        scen_rows.append({
            "Scenario":       scen_name,
            "Median Return":  f"{asset_ret_exp*100:+.1f}%",
            "Median Price":   f"${median:,.2f}",
            "5th Percentile": f"${pct05:,.2f}",
            "95th Percentile":f"${pct95:,.2f}",
        })
        scen_paths[scen_name] = terminal

    return FactorForecastResult(
        ticker           = ticker,
        factors_used     = available,
        alpha_ann        = alpha_ann,
        betas            = betas,
        r_squared        = reg["r_squared"],
        adj_r_squared    = reg["adj_r_squared"],
        t_stats          = {k: v for k, v in reg["t_stats"].items() if k != "alpha"},
        p_values         = {k: v for k, v in reg["p_values"].items() if k != "alpha"},
        residual_vol_ann = resid_vol_ann,
        contribution_pct = contrib,
        scenarios_df     = pd.DataFrame(scen_rows),
        scenario_paths   = scen_paths,
    )
