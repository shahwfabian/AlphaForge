"""
Benchmark comparison module.

Computes alpha, beta, correlation, and excess return vs. a benchmark
(typically SPY or another broad market ETF).
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute CAPM-style benchmark comparison metrics.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark (e.g. SPY).
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.04). Default 0.

    Returns
    -------
    dict
        alpha, beta, r_squared, correlation, information_ratio,
        tracking_error, excess_return.
    """
    # Align on common dates
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 10:
        logger.warning("Insufficient overlapping data for benchmark comparison.")
        return {}

    s = aligned["strategy"]
    b = aligned["benchmark"]
    n = len(s)

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    s_excess = s - daily_rf
    b_excess = b - daily_rf

    # ---- Beta ----
    # β = Cov(r_strategy, r_benchmark) / Var(r_benchmark)
    cov_matrix = np.cov(s.values, b.values, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0

    # ---- Alpha (Jensen's Alpha) ----
    # α = E[r_s] - [rf + β * (E[r_b] - rf)]   (annualised)
    # Annualise daily values
    ann_s = (1 + s.mean()) ** TRADING_DAYS_PER_YEAR - 1
    ann_b = (1 + b.mean()) ** TRADING_DAYS_PER_YEAR - 1
    alpha = ann_s - (risk_free_rate + beta * (ann_b - risk_free_rate))

    # ---- Correlation & R-squared ----
    correlation = s.corr(b)
    r_squared = correlation ** 2

    # ---- Tracking Error ----
    # TE = std(r_strategy - r_benchmark) * sqrt(252)
    active_returns = s - b
    tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ---- Information Ratio ----
    # IR = (E[r_strategy] - E[r_benchmark]) / TE_daily
    te_daily = active_returns.std()
    ir = (active_returns.mean() / te_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
          if te_daily > 0 else 0.0)

    # ---- Annualised Excess Return ----
    ann_strategy = (1 + s.mean()) ** TRADING_DAYS_PER_YEAR - 1
    ann_benchmark = (1 + b.mean()) ** TRADING_DAYS_PER_YEAR - 1
    excess_return = ann_strategy - ann_benchmark

    # ---- Up / Down Capture ----
    up_days = b[b >= 0]
    down_days = b[b < 0]
    up_capture = (
        s[up_days.index].mean() / up_days.mean()
        if len(up_days) > 0 and up_days.mean() != 0 else 0.0
    )
    down_capture = (
        s[down_days.index].mean() / down_days.mean()
        if len(down_days) > 0 and down_days.mean() != 0 else 0.0
    )

    return {
        "Alpha": round(float(alpha), 6),
        "Beta": round(float(beta), 4),
        "R_Squared": round(float(r_squared), 4),
        "Correlation": round(float(correlation), 4),
        "Tracking_Error": round(float(tracking_error), 6),
        "Information_Ratio": round(float(ir), 4),
        "Excess_Return": round(float(excess_return), 6),
        "Up_Capture": round(float(up_capture), 4),
        "Down_Capture": round(float(down_capture), 4),
        "Benchmark_Ann_Return": round(float(ann_benchmark), 6),
    }


def relative_performance(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> pd.DataFrame:
    """
    Produce a DataFrame comparing strategy vs benchmark cumulative returns.

    Parameters
    ----------
    strategy_equity : pd.Series
        Portfolio value series indexed by date.
    benchmark_equity : pd.Series
        Benchmark portfolio value series indexed by date.

    Returns
    -------
    pd.DataFrame
        Columns: Strategy_Cumulative, Benchmark_Cumulative, Active_Return.
    """
    aligned = pd.concat(
        [
            strategy_equity.rename("Strategy_Value"),
            benchmark_equity.rename("Benchmark_Value"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    aligned["Strategy_Cumulative"] = (
        aligned["Strategy_Value"] / aligned["Strategy_Value"].iloc[0] - 1
    )
    aligned["Benchmark_Cumulative"] = (
        aligned["Benchmark_Value"] / aligned["Benchmark_Value"].iloc[0] - 1
    )
    aligned["Active_Return"] = (
        aligned["Strategy_Cumulative"] - aligned["Benchmark_Cumulative"]
    )

    return aligned[["Strategy_Cumulative", "Benchmark_Cumulative", "Active_Return"]]
