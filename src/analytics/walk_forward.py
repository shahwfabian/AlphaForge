"""
Walk-Forward Validation.

Separates in-sample (IS) parameter optimisation from out-of-sample (OOS)
performance evaluation across rolling time windows.

Purpose
-------
- Detect overfitting: IS Sharpe >> OOS Sharpe signals curve-fitting.
- Estimate realistic live performance.
- Provide a degradation ratio: OOS Sharpe / IS Sharpe.

Window modes
------------
Rolling   — fixed-size training window slides forward each split.
Anchored  — training window expands from a fixed start date (IS grows).

Purge gap
---------
An optional gap of ``purge_days`` between the end of training data and
the start of the test window prevents information leakage from overlapping
return calculations or features that look back into the training period.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .performance import compute_all_metrics

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class WalkForwardSplit:
    split_id:         int
    train_start:      str
    train_end:        str
    test_start:       str
    test_end:         str
    best_params:      dict
    is_sharpe:        float
    is_max_dd:        float
    is_ann_return:    float
    oos_sharpe:       float
    oos_max_dd:       float
    oos_ann_return:   float
    oos_cumret:       float
    train_rows:       int
    test_rows:        int


# ── Walk-forward validator ────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    General-purpose walk-forward validator for any registered strategy.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy in the plugin registry.
    param_grid : dict[str, list]
        Parameter search space.
    n_splits : int
        Number of walk-forward windows.
    train_pct : float
        Fraction of each window used for training (default 0.7).
    anchored : bool
        If True, training always starts from the first bar (expanding IS).
    purge_days : int
        Gap between IS end and OOS start (prevents look-forward leakage).
    initial_capital : float
    transaction_cost : float
    slippage : float
    """

    def __init__(
        self,
        strategy_name: str,
        param_grid: dict[str, list[Any]],
        n_splits: int = 3,
        train_pct: float = 0.70,
        anchored: bool = False,
        purge_days: int = 0,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.strategy_name    = strategy_name
        self.param_grid       = param_grid
        self.n_splits         = n_splits
        self.train_pct        = train_pct
        self.anchored         = anchored
        self.purge_days       = purge_days
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage         = slippage

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, ticker: str = "ASSET") -> pd.DataFrame:
        """
        Run walk-forward validation across all splits.

        Parameters
        ----------
        df : pd.DataFrame
            Full preprocessed OHLCV history.
        ticker : str

        Returns
        -------
        pd.DataFrame
            One row per split with IS and OOS metrics.
        """
        from .optimizer import StrategyOptimizer
        from ..strategies.registry import build_strategy
        from ..backtester.engine import BacktestEngine

        total = len(df)
        window = total // self.n_splits
        splits: list[WalkForwardSplit] = []

        for i in range(self.n_splits):
            # Window boundaries
            win_start = 0 if self.anchored else i * window
            win_end   = (i + 1) * window if i < self.n_splits - 1 else total

            window_df    = df.iloc[win_start:win_end]
            split_point  = int(len(window_df) * self.train_pct)

            if split_point < 60 or (len(window_df) - split_point) < 20:
                logger.warning("Split %d too small, skipping.", i)
                continue

            train_end_idx  = win_start + split_point
            test_start_idx = train_end_idx + self.purge_days
            test_end_idx   = win_end

            if test_start_idx >= test_end_idx:
                logger.warning("Split %d: purge_days exhausted test window.", i)
                continue

            train_df = df.iloc[win_start: train_end_idx]
            test_df  = df.iloc[test_start_idx: test_end_idx]

            logger.info(
                "WF split %d | train %s→%s (%d rows) | test %s→%s (%d rows)",
                i,
                train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
                test_df.index[0].date(),  test_df.index[-1].date(),  len(test_df),
            )

            # ── Optimise on training window ──────────────────────────
            opt = StrategyOptimizer(
                self.strategy_name,
                self.param_grid,
                self.initial_capital,
                self.transaction_cost,
                self.slippage,
            )
            is_results = opt.grid_search(train_df, ticker)
            if is_results.empty:
                logger.warning("Split %d: no valid IS results.", i)
                continue

            best_row    = is_results.iloc[0]
            best_params = best_row.get("params", {})

            is_sharpe    = float(best_row.get("Sharpe_Ratio",    0.0))
            is_max_dd    = float(best_row.get("Max_Drawdown",    0.0))
            is_ann_ret   = float(best_row.get("Annualised_Return", 0.0))

            # ── Evaluate on test window ──────────────────────────────
            try:
                strategy = build_strategy(self.strategy_name, best_params)
                engine   = BacktestEngine(
                    strategy, self.initial_capital,
                    self.transaction_cost, self.slippage,
                )
                oos_res     = engine.run(test_df, ticker)
                oos_metrics = compute_all_metrics(oos_res["equity_curve"])
            except Exception as e:
                logger.warning("Split %d OOS evaluation failed: %s", i, e)
                continue

            oos_sharpe  = float(oos_metrics.get("Sharpe_Ratio",    0.0))
            oos_max_dd  = float(oos_metrics.get("Max_Drawdown",    0.0))
            oos_ann_ret = float(oos_metrics.get("Annualised_Return", 0.0))
            oos_cumret  = float(oos_metrics.get("Cumulative_Return", 0.0))

            splits.append(WalkForwardSplit(
                split_id       = i + 1,
                train_start    = str(train_df.index[0].date()),
                train_end      = str(train_df.index[-1].date()),
                test_start     = str(test_df.index[0].date()),
                test_end       = str(test_df.index[-1].date()),
                best_params    = best_params,
                is_sharpe      = round(is_sharpe,  3),
                is_max_dd      = round(is_max_dd,  3),
                is_ann_return  = round(is_ann_ret, 3),
                oos_sharpe     = round(oos_sharpe,  3),
                oos_max_dd     = round(oos_max_dd,  3),
                oos_ann_return = round(oos_ann_ret, 3),
                oos_cumret     = round(oos_cumret,  3),
                train_rows     = len(train_df),
                test_rows      = len(test_df),
            ))

        if not splits:
            return pd.DataFrame()

        return pd.DataFrame([vars(s) for s in splits])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(wf_df: pd.DataFrame) -> dict:
        """Aggregate IS vs OOS stats across all splits."""
        if wf_df.empty:
            return {}

        avg_is  = wf_df["is_sharpe"].mean()
        avg_oos = wf_df["oos_sharpe"].mean()
        ratio   = avg_oos / avg_is if avg_is != 0 else 0.0

        return {
            "Avg_IS_Sharpe":              round(float(avg_is),  3),
            "Avg_OOS_Sharpe":             round(float(avg_oos), 3),
            "Sharpe_Degradation_Ratio":   round(float(ratio),   3),
            "Avg_IS_MaxDD":               round(float(wf_df["is_max_dd"].mean()),  3),
            "Avg_OOS_MaxDD":              round(float(wf_df["oos_max_dd"].mean()), 3),
            "Avg_OOS_CumReturn":          round(float(wf_df["oos_cumret"].mean()), 3),
            "N_Splits":                   len(wf_df),
            "Pct_Splits_OOS_Positive":    round(
                float((wf_df["oos_sharpe"] > 0).mean()), 3
            ),
        }


# ── Convenience wrappers (backward-compatible) ────────────────────────────────

def walk_forward_ma(
    df: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    train_pct: float = 0.70,
    n_splits: int = 3,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Walk-forward validation for Moving Average Crossover."""
    validator = WalkForwardValidator(
        strategy_name="Moving Average Crossover",
        param_grid={"short_window": short_windows, "long_window": long_windows},
        n_splits=n_splits,
        train_pct=train_pct,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    return validator.validate(df)


def walk_forward_generic(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    n_splits: int = 3,
    train_pct: float = 0.70,
    anchored: bool = False,
    purge_days: int = 0,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Walk-forward validation for any registered strategy."""
    validator = WalkForwardValidator(
        strategy_name=strategy_name,
        param_grid=param_grid,
        n_splits=n_splits,
        train_pct=train_pct,
        anchored=anchored,
        purge_days=purge_days,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    return validator.validate(df)


def summarise_walk_forward(wf_df: pd.DataFrame) -> dict:
    """Backward-compatible wrapper around WalkForwardValidator.summarise."""
    return WalkForwardValidator.summarise(wf_df)
