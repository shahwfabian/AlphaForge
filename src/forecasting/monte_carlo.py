"""
AlphaForge — Monte Carlo Probabilistic Forecasting Engine
==========================================================
Generates probability distributions over future market paths using
Geometric Brownian Motion.  This module models *uncertainty*, not
certainty.  Output labelled "Probability Forecast", never "Price Prediction".

Mathematical framework
-----------------------
Continuous-time GBM:  dS = μ S dt + σ S dW

Discretised (Euler-Maruyama, Δt = 1/252):
    S(t+Δt) = S(t) · exp( (μ - σ²/2) Δt  +  σ √Δt · Z )
    Z ~ N(0,1)  i.i.d.

Log-return over horizon T:
    log(S_T / S_0) ~ N( (μ - σ²/2) T,  σ² T )

This gives an *analytical* terminal distribution complementing the
simulation, providing a calibration check.

Disclaimer
----------
Probabilistic forecasts are not predictions of future prices.
They represent possible outcomes under a simplified statistical model
of historical behaviour.  Past distributions do not guarantee future ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Forecasts are probabilistic estimates derived from historical market "
    "behaviour and statistical assumptions (Geometric Brownian Motion). "
    "They do not represent certainty and should not be interpreted as "
    "investment advice. Past distributions do not guarantee future outcomes."
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MCForecastResult:
    """Container for all Monte Carlo forecast outputs."""

    # ── Simulation metadata ────────────────────────────────────────────────────
    ticker: str
    horizon: int                    # trading days
    n_paths: int
    initial_price: float
    drift_ann: float                # annualised μ (historical log-return mean × 252)
    volatility_ann: float           # annualised σ
    drift_adj: float                # μ - σ²/2 (GBM drift adjustment)

    # ── Path data ──────────────────────────────────────────────────────────────
    paths: np.ndarray               # (n_paths, horizon) simulated price paths
    dates: pd.DatetimeIndex         # forecast date index

    # ── Percentile paths ───────────────────────────────────────────────────────
    pct05: pd.Series
    pct10: pd.Series
    pct25: pd.Series
    pct50: pd.Series                # median
    pct75: pd.Series
    pct90: pd.Series
    pct95: pd.Series

    # ── Terminal value distribution ───────────────────────────────────────────
    terminal_values: np.ndarray     # S_T for each path
    terminal_mean: float
    terminal_std: float
    terminal_skew: float
    terminal_kurt: float

    # ── Probability estimates (from simulation) ───────────────────────────────
    prob_above_initial: float       # P(S_T > S_0)
    prob_loss_5pct: float           # P(S_T < 0.95 S_0)
    prob_loss_10pct: float          # P(S_T < 0.90 S_0)
    prob_loss_20pct: float          # P(S_T < 0.80 S_0)
    prob_gain_10pct: float          # P(S_T > 1.10 S_0)
    prob_gain_20pct: float          # P(S_T > 1.20 S_0)

    # ── Analytical terminal CI (lognormal) ────────────────────────────────────
    ci_50_lo: float
    ci_50_hi: float
    ci_68_lo: float
    ci_68_hi: float
    ci_95_lo: float
    ci_95_hi: float

    disclaimer: str = field(default=_DISCLAIMER, repr=False)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Return (lower, upper) empirical CI from simulation paths."""
        alpha = (1.0 - level) / 2.0
        lo = float(np.percentile(self.terminal_values, alpha * 100))
        hi = float(np.percentile(self.terminal_values, (1.0 - alpha) * 100))
        return lo, hi

    def summary(self) -> dict:
        return {
            "Ticker":            self.ticker,
            "Horizon (days)":    self.horizon,
            "Simulations":       self.n_paths,
            "Initial Price":     f"${self.initial_price:,.2f}",
            "Ann. Drift (μ)":    f"{self.drift_ann*100:+.2f}%",
            "Ann. Volatility":   f"{self.volatility_ann*100:.2f}%",
            "Median Terminal":   f"${self.pct50.iloc[-1]:,.2f}",
            "P5  Terminal":      f"${self.pct05.iloc[-1]:,.2f}",
            "P95 Terminal":      f"${self.pct95.iloc[-1]:,.2f}",
            "P(S_T > S_0)":      f"{self.prob_above_initial*100:.1f}%",
            "P(loss > 10%)":     f"{self.prob_loss_10pct*100:.1f}%",
            "P(gain > 10%)":     f"{self.prob_gain_10pct*100:.1f}%",
        }


# ── Core simulation ───────────────────────────────────────────────────────────

def run_gbm_simulation(
    prices: pd.Series,
    horizon: int = 63,
    n_paths: int = 1000,
    ticker: str = "ASSET",
    seed: Optional[int] = 42,
) -> MCForecastResult:
    """
    Simulate future price paths under Geometric Brownian Motion.

    Parameters
    ----------
    prices : pd.Series
        Historical price series (adjusted close).
    horizon : int
        Forecast horizon in trading days (21 = 1 month, 63 = 1 quarter,
        126 = 6 months, 252 = 1 year).
    n_paths : int
        Number of Monte Carlo paths (1_000 / 5_000 / 10_000 recommended).
    ticker : str
        Ticker label for output labelling.
    seed : int | None
        RNG seed for reproducibility.  Pass None for fresh randomness.

    Returns
    -------
    MCForecastResult
    """
    if len(prices) < 30:
        raise ValueError("Need at least 30 historical prices for GBM calibration.")

    rng = np.random.default_rng(seed)

    # ── Parameter estimation ──────────────────────────────────────────────────
    log_rets = np.log(prices / prices.shift(1)).dropna().values

    mu_daily    = float(log_rets.mean())
    sigma_daily = float(log_rets.std(ddof=1))

    mu_ann      = mu_daily    * 252.0
    sigma_ann   = sigma_daily * np.sqrt(252.0)
    drift_adj   = mu_daily - 0.5 * sigma_daily ** 2   # daily GBM drift

    S0 = float(prices.iloc[-1])

    # ── GBM path generation ───────────────────────────────────────────────────
    # Z: (n_paths, horizon)  i.i.d. standard normals
    Z           = rng.standard_normal((n_paths, horizon))
    increments  = np.exp(drift_adj + sigma_daily * Z)
    cum_product = np.cumprod(increments, axis=1)       # (n_paths, horizon)
    paths       = S0 * cum_product

    # ── Date index ────────────────────────────────────────────────────────────
    last_date = prices.index[-1] if isinstance(prices.index, pd.DatetimeIndex) \
                else pd.Timestamp.today()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    # ── Percentile paths ──────────────────────────────────────────────────────
    def _pct(p: float) -> pd.Series:
        return pd.Series(np.percentile(paths, p, axis=0), index=future_dates)

    # ── Terminal distribution ─────────────────────────────────────────────────
    terminal = paths[:, -1]

    # ── Analytical lognormal CIs ──────────────────────────────────────────────
    T_years     = horizon / 252.0
    ln_mean     = np.log(S0) + (mu_ann - 0.5 * sigma_ann ** 2) * T_years
    ln_std      = sigma_ann * np.sqrt(T_years)

    def _analytical_ci(level: float) -> tuple[float, float]:
        alpha = (1.0 - level) / 2.0
        lo = float(np.exp(stats.norm.ppf(alpha,    ln_mean, ln_std)))
        hi = float(np.exp(stats.norm.ppf(1 - alpha, ln_mean, ln_std)))
        return lo, hi

    ci50 = _analytical_ci(0.50)
    ci68 = _analytical_ci(0.68)
    ci95 = _analytical_ci(0.95)

    return MCForecastResult(
        ticker           = ticker,
        horizon          = horizon,
        n_paths          = n_paths,
        initial_price    = S0,
        drift_ann        = mu_ann,
        volatility_ann   = sigma_ann,
        drift_adj        = drift_adj * 252.0,  # annualised for display

        paths            = paths,
        dates            = future_dates,

        pct05  = _pct(5),
        pct10  = _pct(10),
        pct25  = _pct(25),
        pct50  = _pct(50),
        pct75  = _pct(75),
        pct90  = _pct(90),
        pct95  = _pct(95),

        terminal_values  = terminal,
        terminal_mean    = float(terminal.mean()),
        terminal_std     = float(terminal.std()),
        terminal_skew    = float(stats.skew(terminal)),
        terminal_kurt    = float(stats.kurtosis(terminal)),

        prob_above_initial = float(np.mean(terminal >  S0)),
        prob_loss_5pct     = float(np.mean(terminal <  0.95 * S0)),
        prob_loss_10pct    = float(np.mean(terminal <  0.90 * S0)),
        prob_loss_20pct    = float(np.mean(terminal <  0.80 * S0)),
        prob_gain_10pct    = float(np.mean(terminal >  1.10 * S0)),
        prob_gain_20pct    = float(np.mean(terminal >  1.20 * S0)),

        ci_50_lo = ci50[0], ci_50_hi = ci50[1],
        ci_68_lo = ci68[0], ci_68_hi = ci68[1],
        ci_95_lo = ci95[0], ci_95_hi = ci95[1],
    )


# ── Horizon presets ───────────────────────────────────────────────────────────

HORIZON_PRESETS: dict[str, int] = {
    "1 Month  (21 days)":   21,
    "3 Months (63 days)":   63,
    "6 Months (126 days)": 126,
    "1 Year   (252 days)": 252,
}

N_PATHS_PRESETS: list[int] = [1_000, 5_000, 10_000]
