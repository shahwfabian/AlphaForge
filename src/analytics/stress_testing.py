"""
AlphaForge — Stress Testing
============================
Applies predefined and custom market shock scenarios to a strategy's
historical return series to estimate potential portfolio impact under
adverse conditions.

Methodology note
----------------
Impact estimates use a strategy-beta proxy scaled from historical
volatility relative to a typical equity market volatility of 16%.
These are rough approximations, not precise risk model outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Predefined scenarios ──────────────────────────────────────────────────────

STRESS_SCENARIOS: dict[str, dict] = {
    "2008 Global Financial Crisis": {
        "description": "S&P 500 fell ~38% in 2008; Oct alone −17%; realised vol 5× normal",
        "market_shock_pct": -0.38,
        "vol_multiplier":    5.0,
        "duration_days":    252,
    },
    "COVID-19 Crash (Feb–Mar 2020)": {
        "description": "S&P 500 dropped −34% in 33 calendar days — fastest bear market ever",
        "market_shock_pct": -0.34,
        "vol_multiplier":    4.0,
        "duration_days":     33,
    },
    "Dot-Com Bust (2000–2002)": {
        "description": "NASDAQ −78% over ~2.5 years; S&P 500 −49% peak-to-trough",
        "market_shock_pct": -0.49,
        "vol_multiplier":    2.5,
        "duration_days":    504,
    },
    "2022 Rate-Hike Shock": {
        "description": "Fed raised rates 425bps; S&P 500 −19.4%; AGG bonds −13% in 2022",
        "market_shock_pct": -0.20,
        "vol_multiplier":    2.0,
        "duration_days":    252,
    },
    "Tech Selloff (−25%)": {
        "description": "Sector rotation out of technology / growth stocks",
        "market_shock_pct": -0.25,
        "vol_multiplier":    2.5,
        "duration_days":     45,
    },
    "Inflation Shock": {
        "description": "Persistent 8%+ CPI eroding real returns and compressing multiples",
        "market_shock_pct": -0.15,
        "vol_multiplier":    1.6,
        "duration_days":    126,
    },
    "Rising Rate / Bond Shock": {
        "description": "100 bps parallel rate rise; long-duration assets hurt most",
        "market_shock_pct": -0.12,
        "vol_multiplier":    1.8,
        "duration_days":     40,
    },
    "Flash Crash (Single Day −5%)": {
        "description": "Sudden single-session gap-down without macro catalyst",
        "market_shock_pct": -0.05,
        "vol_multiplier":    3.0,
        "duration_days":      1,
    },
    "Mild Correction (−10%)": {
        "description": "Typical healthy 10% market pullback",
        "market_shock_pct": -0.10,
        "vol_multiplier":    1.5,
        "duration_days":     20,
    },
}


# ── Core calculation ──────────────────────────────────────────────────────────

def _estimate_beta_proxy(returns: pd.Series, market_vol: float = 0.16) -> float:
    """
    Rough beta estimate: ratio of strategy vol to typical market vol.
    Clamped to [0.05, 2.5].
    """
    strategy_vol = float(returns.std()) * np.sqrt(252)
    if strategy_vol == 0:
        return 1.0
    return float(np.clip(strategy_vol / market_vol, 0.05, 2.5))


def run_stress_scenarios(
    returns: pd.Series,
    initial_value: float = 100_000.0,
    scenarios: dict | None = None,
) -> pd.DataFrame:
    """
    Estimate portfolio impact for every stress scenario.

    For each scenario the portfolio loss is approximated as:
        estimated_loss_pct ≈ market_shock_pct × beta_proxy

    This is a conservative educational estimate, not a precise VaR calculation.

    Parameters
    ----------
    returns : pd.Series
        Historical daily strategy returns.
    initial_value : float
        Portfolio value at start of each shock (e.g. current NAV).
    scenarios : dict | None
        Custom scenario dict or ``None`` to use ``STRESS_SCENARIOS``.

    Returns
    -------
    pd.DataFrame
        Columns: Severity_Rank, Scenario, Description, Market_Shock_Pct,
        Duration_Days, Estimated_Loss_Pct, Estimated_Loss_USD, Ending_Value.
        Sorted worst-to-best by Estimated_Loss_Pct.
    """
    scen        = scenarios if scenarios is not None else STRESS_SCENARIOS
    beta_proxy  = _estimate_beta_proxy(returns)
    rows: list[dict] = []

    for name, params in scen.items():
        shock       = float(params.get("market_shock_pct", -0.10))
        loss_pct    = float(np.clip(shock * beta_proxy, -0.95, 0.50))
        loss_usd    = initial_value * loss_pct
        ending_val  = initial_value + loss_usd

        rows.append({
            "Scenario":           name,
            "Description":        params.get("description", ""),
            "Market_Shock_Pct":   shock,
            "Duration_Days":      int(params.get("duration_days", 21)),
            "Vol_Multiplier":     float(params.get("vol_multiplier", 1.0)),
            "Estimated_Loss_Pct": loss_pct,
            "Estimated_Loss_USD": loss_usd,
            "Ending_Value":       ending_val,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Estimated_Loss_Pct").reset_index(drop=True)
        df.insert(0, "Severity_Rank", range(1, len(df) + 1))

    return df


def custom_scenario(
    returns: pd.Series,
    initial_value: float,
    market_shock_pct: float,
    duration_days: int = 21,
    vol_multiplier: float = 1.0,
) -> dict:
    """
    Run a single custom shock scenario.

    Parameters
    ----------
    returns : pd.Series
        Historical daily strategy returns.
    initial_value : float
    market_shock_pct : float
        Expected market decline (negative, e.g. -0.20 for −20%).
    duration_days : int
    vol_multiplier : float

    Returns
    -------
    dict
        estimated_loss_pct, estimated_loss_usd, ending_value, beta_proxy.
    """
    beta     = _estimate_beta_proxy(returns)
    loss_pct = float(np.clip(market_shock_pct * beta, -0.95, 0.50))
    loss_usd = initial_value * loss_pct

    return {
        "market_shock_pct":   market_shock_pct,
        "duration_days":      duration_days,
        "vol_multiplier":     vol_multiplier,
        "beta_proxy":         beta,
        "estimated_loss_pct": loss_pct,
        "estimated_loss_usd": loss_usd,
        "ending_value":       initial_value + loss_usd,
    }
