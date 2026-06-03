"""
AlphaForge — Forecast Confidence Interval Engine
==================================================
Constructs and reports confidence intervals for probabilistic forecasts.

Two complementary approaches
------------------------------
1. Parametric (analytical):
   For GBM, log(S_T / S_0) ~ N(μ_adj · T, σ² · T) — exact CI bands.

2. Empirical (bootstrap):
   Percentile bands derived directly from Monte Carlo path ensemble.
   No distributional assumptions beyond the MC model.

Coverage levels: 50%, 68% (≈1σ), 90%, 95%

Every interval is expressed as:
  "There is a X% probability that the asset lies between $L and $U
   under the stated model assumptions."

IMPORTANT: These intervals are MODEL-DEPENDENT — they are only as valid
as the underlying GBM assumptions (constant drift and volatility, no
jumps, no regime changes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .monte_carlo import MCForecastResult

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Confidence intervals are model-dependent.  They assume constant drift "
    "and volatility (Geometric Brownian Motion).  Real markets exhibit "
    "fat tails, volatility clustering, and regime changes that invalidate "
    "these bounds.  Treat them as indicative ranges, not guarantees."
)

COVERAGE_LEVELS: list[float] = [0.50, 0.68, 0.90, 0.95]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CIBand:
    level: float
    lo:    pd.Series
    hi:    pd.Series
    label: str

    def terminal(self) -> tuple[float, float]:
        return float(self.lo.iloc[-1]), float(self.hi.iloc[-1])


@dataclass
class ConfidenceResult:
    ticker:          str
    initial_price:   float
    horizon:         int
    dates:           pd.DatetimeIndex
    median_path:     pd.Series

    # Empirical bands (from MC percentiles)
    empirical: list[CIBand]

    # Analytical bands (GBM closed-form)
    analytical: list[CIBand]

    # Summary table
    terminal_summary: pd.DataFrame

    disclaimer: str = field(default=_DISCLAIMER, repr=False)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_confidence_intervals(result: MCForecastResult) -> ConfidenceResult:
    """
    Construct empirical + analytical CI bands from an MCForecastResult.

    Parameters
    ----------
    result : MCForecastResult
        Output of run_gbm_simulation().

    Returns
    -------
    ConfidenceResult
    """
    S0      = result.initial_price
    horizon = result.horizon
    mu_ann  = result.drift_ann
    sig_ann = result.volatility_ann
    dates   = result.dates

    # ── Empirical bands ────────────────────────────────────────────────────────
    pct_map = {
        0.50: (25, 75),
        0.68: (16, 84),
        0.90: (5,  95),
        0.95: (2.5, 97.5),
    }
    empirical: list[CIBand] = []
    for level, (lo_p, hi_p) in pct_map.items():
        lo = pd.Series(np.percentile(result.paths, lo_p, axis=0), index=dates)
        hi = pd.Series(np.percentile(result.paths, hi_p, axis=0), index=dates)
        empirical.append(CIBand(
            level = level,
            lo    = lo,
            hi    = hi,
            label = f"{int(level*100)}% Empirical CI",
        ))

    # ── Analytical bands (GBM lognormal) ──────────────────────────────────────
    mu_daily  = mu_ann  / 252.0
    sig_daily = sig_ann / np.sqrt(252.0)
    drift_adj = mu_daily - 0.5 * sig_daily ** 2

    analytical: list[CIBand] = []
    for level in COVERAGE_LEVELS:
        alpha  = (1.0 - level) / 2.0
        z_lo   = stats.norm.ppf(alpha)
        z_hi   = stats.norm.ppf(1.0 - alpha)
        t_arr  = np.arange(1, horizon + 1)
        lo_arr = S0 * np.exp(drift_adj * t_arr + sig_daily * z_lo * np.sqrt(t_arr))
        hi_arr = S0 * np.exp(drift_adj * t_arr + sig_daily * z_hi * np.sqrt(t_arr))
        analytical.append(CIBand(
            level = level,
            lo    = pd.Series(lo_arr, index=dates),
            hi    = pd.Series(hi_arr, index=dates),
            label = f"{int(level*100)}% Analytical CI",
        ))

    # ── Terminal summary table ─────────────────────────────────────────────────
    rows = []
    for emp, ana in zip(empirical, analytical):
        e_lo, e_hi = emp.terminal()
        a_lo, a_hi = ana.terminal()
        rows.append({
            "Coverage":         f"{int(emp.level*100)}%",
            "Empirical Lower":  f"${e_lo:,.2f}",
            "Empirical Upper":  f"${e_hi:,.2f}",
            "Analytical Lower": f"${a_lo:,.2f}",
            "Analytical Upper": f"${a_hi:,.2f}",
        })
    summary_df = pd.DataFrame(rows)

    return ConfidenceResult(
        ticker           = result.ticker,
        initial_price    = S0,
        horizon          = horizon,
        dates            = dates,
        median_path      = result.pct50,
        empirical        = empirical,
        analytical       = analytical,
        terminal_summary = summary_df,
    )


# ── Standalone probability query ──────────────────────────────────────────────

def probability_in_range(
    result: MCForecastResult,
    low: float,
    high: float,
    at_step: Optional[int] = None,  # type: ignore[name-defined]
) -> float:
    """
    Estimate P(low < S_t < high) empirically from MC paths.

    Parameters
    ----------
    result : MCForecastResult
    low, high : price bounds
    at_step : day index (1-based); defaults to horizon end

    Returns
    -------
    float probability in [0, 1]
    """
    from typing import Optional  # local import to avoid circular
    col = (at_step - 1) if at_step else result.horizon - 1
    col = min(max(col, 0), result.horizon - 1)
    terminal = result.paths[:, col]
    return float(np.mean((terminal > low) & (terminal < high)))


def expected_terminal(result: MCForecastResult) -> float:
    """E[S_T] from simulation."""
    return float(result.terminal_values.mean())
