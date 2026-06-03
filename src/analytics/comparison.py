"""
AlphaForge — Multi-Strategy Comparison Analytics
=================================================
Helpers for running, ranking, and formatting side-by-side
strategy comparisons.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Metrics shown in the comparison table (in display order)
COMPARISON_METRICS = [
    "Cumulative_Return",
    "CAGR",
    "Sharpe_Ratio",
    "Sortino_Ratio",
    "Max_Drawdown",
    "Calmar_Ratio",
    "Win_Rate",
    "Profit_Factor",
    "Volatility_Ann",
    "VaR_95_Historical",
]

# Only Volatility_Ann is lower-is-better (positive number, lower = less risky).
# Max_Drawdown and VaR are negative — higher (less negative) is already better,
# so they sort descending with everything else.
_LOWER_IS_BETTER = {"Volatility_Ann"}


def build_comparison_table(metrics_map: dict[str, dict]) -> pd.DataFrame:
    """
    Assemble a raw comparison DataFrame from a strategy→metrics mapping.

    Parameters
    ----------
    metrics_map : dict[str, dict]
        {"Strategy Name": {metric_key: value}, ...}

    Returns
    -------
    pd.DataFrame
        Rows = strategies, columns = COMPARISON_METRICS.
    """
    if not metrics_map:
        return pd.DataFrame(columns=COMPARISON_METRICS)
    rows = []
    for name, m in metrics_map.items():
        row = {"Strategy": name}
        for k in COMPARISON_METRICS:
            row[k] = m.get(k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


def rank_strategies(
    metrics_map: dict[str, dict],
    by: str = "Sharpe_Ratio",
) -> pd.DataFrame:
    """
    Rank strategies by a chosen metric and return the comparison table
    with a Rank column prepended.

    Parameters
    ----------
    metrics_map : dict
    by : str
        Column to sort by.  Defaults to Sharpe_Ratio.

    Returns
    -------
    pd.DataFrame
        Ranked comparison table (raw values, not formatted).
    """
    df = build_comparison_table(metrics_map)
    ascending = by in _LOWER_IS_BETTER
    if by in df.columns:
        df = df.sort_values(by, ascending=ascending, na_position="last")
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply human-readable formatting to a raw comparison DataFrame.

    Percentage columns become "+12.34%" strings; ratio columns become "1.234".
    """
    fmt = df.copy().astype(object)

    pct_cols   = {"Cumulative_Return", "CAGR", "Max_Drawdown",
                  "Win_Rate", "VaR_95_Historical", "Volatility_Ann"}
    ratio_cols = {"Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio", "Profit_Factor"}

    for col in fmt.columns:
        if col == "Rank":
            continue
        for i, val in enumerate(fmt[col]):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                fmt.at[fmt.index[i], col] = "N/A"
            elif col in pct_cols:
                fmt.at[fmt.index[i], col] = f"{val*100:+.2f}%"
            elif col in ratio_cols:
                fmt.at[fmt.index[i], col] = f"{val:.3f}"
            else:
                fmt.at[fmt.index[i], col] = f"{val:.3f}"

    return fmt


def best_strategy(metrics_map: dict[str, dict], by: str = "Sharpe_Ratio") -> str | None:
    """Return the name of the top-ranked strategy, or None if empty."""
    ranked = rank_strategies(metrics_map, by=by)
    if ranked.empty:
        return None
    return str(ranked.index[0])
