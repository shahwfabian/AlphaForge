"""
AlphaForge — Forward Testing (Out-of-Sample Validation)
========================================================
Implements a clean train / test split workflow:

  1. Split the full data chronologically (no look-ahead).
  2. Evaluate the user-supplied parameters on both IS and OOS windows.
  3. Compute a degradation ratio (OOS Sharpe / IS Sharpe).
  4. Flag potential overfitting when OOS performance is materially worse.

This module is intentionally simple — it evaluates *given* parameters
rather than optimising them, so users understand the distinction between
in-sample fitting and out-of-sample performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ForwardTestResult:
    """Structured result from a single train/test evaluation."""

    # Split metadata
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    train_pct:   float
    best_params: dict = field(default_factory=dict)

    # In-sample metrics
    is_sharpe:  float = 0.0
    is_cagr:    float = 0.0
    is_max_dd:  float = 0.0

    # Out-of-sample metrics
    oos_sharpe:  float = 0.0
    oos_cagr:    float = 0.0
    oos_max_dd:  float = 0.0
    oos_cum_ret: float = 0.0

    # Derived
    degradation_ratio: float = float("nan")
    overfitting_flag:  bool  = False

    def to_dict(self) -> dict:
        return {
            "Train Period":     f"{self.train_start} → {self.train_end}",
            "Test Period":      f"{self.test_start} → {self.test_end}",
            "Train Fraction":   f"{self.train_pct*100:.0f}%",
            "IS Sharpe":        round(self.is_sharpe,  3),
            "IS CAGR":          f"{self.is_cagr*100:+.2f}%",
            "IS Max Drawdown":  f"{self.is_max_dd*100:.2f}%",
            "OOS Sharpe":       round(self.oos_sharpe, 3),
            "OOS CAGR":         f"{self.oos_cagr*100:+.2f}%",
            "OOS Max Drawdown": f"{self.oos_max_dd*100:.2f}%",
            "OOS Cum Return":   f"{self.oos_cum_ret*100:+.2f}%",
            "Degradation Ratio":f"{self.degradation_ratio:.3f}" if not np.isnan(self.degradation_ratio) else "N/A",
            "Overfitting Flag": "⚠️ Yes" if self.overfitting_flag else "✅ No",
        }


# ── Split utility ─────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_pct: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a chronologically ordered DataFrame into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full pre-processed OHLCV data.
    train_pct : float
        Fraction allocated to training (default 0.70 → 70/30 split).

    Returns
    -------
    (train_df, test_df)

    Raises
    ------
    ValueError
        If either partition is too small to be meaningful.
    """
    if not 0.10 <= train_pct <= 0.95:
        raise ValueError(f"train_pct must be in [0.10, 0.95], got {train_pct}.")

    n     = len(df)
    split = int(n * train_pct)

    if split < 30:
        raise ValueError(
            f"Training set has only {split} bars (need ≥30). "
            "Extend the date range or increase train_pct."
        )
    if n - split < 20:
        raise ValueError(
            f"Test set has only {n - split} bars (need ≥20). "
            "Extend the date range or decrease train_pct."
        )

    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_forward_test(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    capital: float        = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float       = 0.0005,
    train_pct: float      = 0.70,
    risk_free_rate: float = 0.0,
    ticker: str           = "ASSET",
) -> ForwardTestResult:
    """
    Evaluate ``params`` on both IS and OOS windows and return a
    ``ForwardTestResult``.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed OHLCV DataFrame.
    strategy_name : str
        Name registered in the strategy registry.
    params : dict
        Strategy parameters to evaluate (not optimised here).
    capital, transaction_cost, slippage : float
    train_pct : float
    risk_free_rate : float
    ticker : str

    Returns
    -------
    ForwardTestResult
    """
    from src.backtester.engine import BacktestEngine
    from src.strategies.registry import build_strategy
    from src.analytics.performance import compute_all_metrics

    train_df, test_df = split_data(df, train_pct)

    def _run(data: pd.DataFrame, label: str) -> dict:
        try:
            strategy = build_strategy(strategy_name, params)
            engine   = BacktestEngine(
                strategy=strategy,
                initial_capital=capital,
                transaction_cost=transaction_cost,
                slippage=slippage,
            )
            res = engine.run(data, ticker)
            return compute_all_metrics(res["equity_curve"], risk_free_rate)
        except Exception as exc:
            logger.warning("Forward test (%s) failed: %s", label, exc)
            return {}

    is_m  = _run(train_df, "IS")
    oos_m = _run(test_df,  "OOS")

    is_sharpe  = float(is_m.get("Sharpe_Ratio",  0.0) or 0.0)
    oos_sharpe = float(oos_m.get("Sharpe_Ratio", 0.0) or 0.0)
    degrad     = oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
    overfit    = (degrad < 0.50) if not np.isnan(degrad) else False

    return ForwardTestResult(
        train_start       = str(train_df.index[0].date()),
        train_end         = str(train_df.index[-1].date()),
        test_start        = str(test_df.index[0].date()),
        test_end          = str(test_df.index[-1].date()),
        train_pct         = train_pct,
        best_params       = params,
        is_sharpe         = is_sharpe,
        is_cagr           = float(is_m.get("CAGR",          0.0) or 0.0),
        is_max_dd         = float(is_m.get("Max_Drawdown",   0.0) or 0.0),
        oos_sharpe        = oos_sharpe,
        oos_cagr          = float(oos_m.get("CAGR",         0.0) or 0.0),
        oos_max_dd        = float(oos_m.get("Max_Drawdown",  0.0) or 0.0),
        oos_cum_ret       = float(oos_m.get("Cumulative_Return", 0.0) or 0.0),
        degradation_ratio = float(degrad),
        overfitting_flag  = overfit,
    )
