"""
AlphaForge — Portfolio Allocation Builder
==========================================
Utilities for constructing, validating, and analysing multi-asset
portfolio weights and returns.

All weight dicts map ticker → float (0 to 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Weight construction ────────────────────────────────────────────────────────

def equal_weight_allocation(symbols: list[str]) -> dict[str, float]:
    """
    Assign equal weight to every symbol.

    >>> equal_weight_allocation(["AAPL", "MSFT", "SPY"])
    {'AAPL': 0.333..., 'MSFT': 0.333..., 'SPY': 0.333...}
    """
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: round(w, 8) for s in symbols}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Rescale weights to sum exactly to 1.0.

    Negative weights are clamped to 0 before normalisation.
    If all weights are 0 or empty, falls back to equal-weight.
    """
    if not weights:
        return {}
    clipped = {k: max(v, 0.0) for k, v in weights.items()}
    total = sum(clipped.values())
    if total == 0:
        return equal_weight_allocation(list(weights.keys()))
    return {k: v / total for k, v in clipped.items()}


def validate_weights(weights: dict[str, float]) -> tuple[bool, str]:
    """
    Validate that a weight dict is a legal portfolio allocation.

    Returns
    -------
    (True, "OK")  if valid
    (False, msg)  if invalid, with a human-readable reason.
    """
    if not weights:
        return False, "No weights provided."
    for sym, w in weights.items():
        if not isinstance(w, (int, float)):
            return False, f"Weight for '{sym}' must be numeric."
        if w < 0:
            return False, f"Weight for '{sym}' is negative ({w:.4f})."
    total = sum(weights.values())
    if abs(total - 1.0) > 0.015:          # allow 1.5% tolerance for rounding
        return False, f"Weights sum to {total:.4f} — must be 1.00."
    return True, "OK"


# ── Return calculations ────────────────────────────────────────────────────────

def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute portfolio daily returns as the weighted sum of asset returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Columns = tickers, index = dates, values = daily returns.
        Missing values are filled with 0.
    weights : dict[str, float]
        Ticker → weight mapping.  Will be normalised to available tickers.

    Returns
    -------
    pd.Series
        Portfolio daily returns, named "Portfolio".

    Raises
    ------
    ValueError
        If none of the weight tickers appear in ``asset_returns``.
    """
    available = {k: v for k, v in weights.items() if k in asset_returns.columns}
    if not available:
        raise ValueError(
            "None of the weight tickers are present in asset_returns. "
            f"Weights: {list(weights)}, Columns: {list(asset_returns.columns)}"
        )
    norm = normalize_weights(available)
    port = pd.Series(0.0, index=asset_returns.index)
    for ticker, w in norm.items():
        port += asset_returns[ticker].fillna(0.0) * w
    return port.rename("Portfolio")


def cumulative_portfolio_value(
    portfolio_returns: pd.Series,
    initial_value: float = 100_000.0,
) -> pd.Series:
    """Compound daily returns into a portfolio value series."""
    return initial_value * (1 + portfolio_returns).cumprod()


# ── Summary helpers ────────────────────────────────────────────────────────────

def allocation_summary(weights: dict[str, float]) -> pd.DataFrame:
    """
    Return a tidy DataFrame of normalised weights, sorted largest-first.

    Columns: Symbol, Weight, Weight_Pct.
    """
    if not weights:
        return pd.DataFrame(columns=["Symbol", "Weight", "Weight_Pct"])
    norm = normalize_weights(weights)
    rows = [
        {"Symbol": sym, "Weight": w, "Weight_Pct": round(w * 100, 2)}
        for sym, w in sorted(norm.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def dominant_asset_warning(weights: dict[str, float], threshold: float = 0.60) -> str | None:
    """
    Return a warning string if any single asset exceeds ``threshold`` weight,
    otherwise return None.
    """
    if not weights:
        return None
    norm = normalize_weights(weights)
    for sym, w in norm.items():
        if w >= threshold:
            return (
                f"⚠️ **{sym}** dominates the portfolio at {w*100:.1f}% weight. "
                "Consider diversifying."
            )
    return None
