"""
Parallel backtesting engine.

Supports three execution patterns:

1. Parameter sweep     — one strategy, many parameter combinations, one asset.
2. Multi-asset sweep   — one strategy + params, many assets.
3. Strategy sweep      — many strategies, one asset.

Workers run via concurrent.futures — ThreadPoolExecutor by default
(safe in all environments, incl. Streamlit), with optional
ProcessPoolExecutor for true CPU parallelism on large grids.

Design notes
------------
- Worker functions are module-level callables (required for pickling).
- All arguments are plain Python types (dict, str, float, DataFrame).
- Results are collected into a tidy pandas DataFrame for downstream use.
- Failed tasks are captured as errors, not raised, so the sweep continues.
"""

import itertools
import logging
import random
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Top-level worker (must be module-level for pickling) ──────────────────────

def _worker(task: dict) -> dict:
    """
    Execute a single backtest task in an isolated worker.

    Parameters
    ----------
    task : dict
        Keys: df, ticker, strategy_name, params,
              initial_capital, transaction_cost, slippage, vol_target.

    Returns
    -------
    dict
        strategy, params, ticker, all performance metrics, error (if any).
    """
    import sys
    from pathlib import Path

    # Ensure src is on path (needed when spawned as a subprocess)
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from src.strategies.registry import build_strategy
        from src.backtester.engine import BacktestEngine
        from src.analytics.performance import compute_all_metrics
        from src.analytics.risk import compute_risk_metrics

        df = task["df"]
        strategy = build_strategy(task["strategy_name"], task["params"])
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=task.get("initial_capital", 100_000.0),
            transaction_cost=task.get("transaction_cost", 0.001),
            slippage=task.get("slippage", 0.0005),
            vol_target=task.get("vol_target", None),
        )
        result = engine.run(df, task["ticker"])
        eq = result["equity_curve"]
        rets = eq["Daily_Return"].dropna()

        perf = compute_all_metrics(eq)
        risk = compute_risk_metrics(rets)

        return {
            "ticker":   task["ticker"],
            "strategy": task["strategy_name"],
            "params":   task["params"],
            "error":    None,
            **perf,
            **risk,
        }

    except Exception as exc:
        logger.warning(
            "Worker failed — strategy=%s params=%s error=%s",
            task.get("strategy_name"), task.get("params"), exc,
        )
        return {
            "ticker":   task.get("ticker", ""),
            "strategy": task.get("strategy_name", ""),
            "params":   task.get("params", {}),
            "error":    str(exc),
        }


# ── Parallel backtester ───────────────────────────────────────────────────────

class ParallelBacktester:
    """
    Concurrent backtesting runner supporting parameter and asset sweeps.

    Parameters
    ----------
    initial_capital : float
    transaction_cost : float
    slippage : float
    vol_target : float or None
    executor : "thread" or "process"
        "thread"  — ThreadPoolExecutor (safe everywhere, incl. Streamlit).
        "process" — ProcessPoolExecutor (true CPU parallelism; requires
                    the calling code to be inside if __name__ == '__main__').
    n_workers : int
        Number of concurrent workers.  Defaults to min(8, cpu_count).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        vol_target: float | None = None,
        executor: Literal["thread", "process"] = "thread",
        n_workers: int | None = None,
    ) -> None:
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage         = slippage
        self.vol_target       = vol_target
        self.executor_type    = executor

        import os
        self.n_workers = n_workers or min(8, os.cpu_count() or 4)

    # ------------------------------------------------------------------
    # Executor factory
    # ------------------------------------------------------------------

    def _executor(self):
        if self.executor_type == "process":
            return ProcessPoolExecutor(max_workers=self.n_workers)
        return ThreadPoolExecutor(max_workers=self.n_workers)

    # ------------------------------------------------------------------
    # Task submission helper
    # ------------------------------------------------------------------

    def _submit_tasks(self, tasks: list[dict]) -> pd.DataFrame:
        """
        Submit all tasks to the executor and collect results.

        Returns a tidy DataFrame sorted by Sharpe_Ratio descending.
        """
        results = []
        logger.info(
            "Submitting %d tasks across %d %s workers.",
            len(tasks), self.n_workers, self.executor_type,
        )

        with self._executor() as exc:
            futures = {exc.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        if "Sharpe_Ratio" in df.columns:
            df.sort_values("Sharpe_Ratio", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # 1. Parameter sweep
    # ------------------------------------------------------------------

    def run_parameter_sweep(
        self,
        df: pd.DataFrame,
        ticker: str,
        strategy_name: str,
        param_grid: dict[str, list],
        max_combinations: int | None = None,
        randomized: bool = False,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Exhaustive or randomized grid search over ``param_grid``.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed OHLCV data.
        ticker : str
        strategy_name : str
        param_grid : dict[str, list]
            e.g. {"short_window": [5, 10, 20], "long_window": [30, 50, 100]}
        max_combinations : int or None
            Cap the number of evaluated combinations.
        randomized : bool
            If True, randomly sample from the grid instead of exhaustive search.
        random_seed : int

        Returns
        -------
        pd.DataFrame
            One row per combination, sorted by Sharpe_Ratio descending.
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        param_dicts = [dict(zip(keys, combo)) for combo in combos]

        if randomized:
            rng = random.Random(random_seed)
            rng.shuffle(param_dicts)

        if max_combinations:
            param_dicts = param_dicts[:max_combinations]

        tasks = [
            {
                "df":               df,
                "ticker":           ticker,
                "strategy_name":    strategy_name,
                "params":           p,
                "initial_capital":  self.initial_capital,
                "transaction_cost": self.transaction_cost,
                "slippage":         self.slippage,
                "vol_target":       self.vol_target,
            }
            for p in param_dicts
        ]
        logger.info(
            "Parameter sweep: %s | %d combinations | randomized=%s",
            strategy_name, len(tasks), randomized,
        )
        return self._submit_tasks(tasks)

    # ------------------------------------------------------------------
    # 2. Multi-asset sweep
    # ------------------------------------------------------------------

    def run_multi_asset(
        self,
        datasets: dict[str, pd.DataFrame],
        strategy_name: str,
        params: dict,
    ) -> pd.DataFrame:
        """
        Run the same strategy + params across multiple assets in parallel.

        Parameters
        ----------
        datasets : dict[ticker → pd.DataFrame]
            Pre-processed DataFrames, one per asset.
        strategy_name : str
        params : dict

        Returns
        -------
        pd.DataFrame
            One row per asset, sorted by Sharpe_Ratio.
        """
        tasks = [
            {
                "df":               df,
                "ticker":           ticker,
                "strategy_name":    strategy_name,
                "params":           params,
                "initial_capital":  self.initial_capital,
                "transaction_cost": self.transaction_cost,
                "slippage":         self.slippage,
                "vol_target":       self.vol_target,
            }
            for ticker, df in datasets.items()
        ]
        logger.info(
            "Multi-asset sweep: %s on %d assets.", strategy_name, len(tasks)
        )
        return self._submit_tasks(tasks)

    # ------------------------------------------------------------------
    # 3. Strategy sweep
    # ------------------------------------------------------------------

    def run_strategy_sweep(
        self,
        df: pd.DataFrame,
        ticker: str,
        strategies: list[tuple[str, dict]],
    ) -> pd.DataFrame:
        """
        Run multiple strategies (each with its own params) on one asset.

        Parameters
        ----------
        df : pd.DataFrame
        ticker : str
        strategies : list of (strategy_name, params) tuples

        Returns
        -------
        pd.DataFrame
        """
        tasks = [
            {
                "df":               df,
                "ticker":           ticker,
                "strategy_name":    name,
                "params":           params,
                "initial_capital":  self.initial_capital,
                "transaction_cost": self.transaction_cost,
                "slippage":         self.slippage,
                "vol_target":       self.vol_target,
            }
            for name, params in strategies
        ]
        logger.info(
            "Strategy sweep: %d strategies on %s.", len(tasks), ticker
        )
        return self._submit_tasks(tasks)

    # ------------------------------------------------------------------
    # Convenience: top-N results
    # ------------------------------------------------------------------

    @staticmethod
    def top_n(
        results_df: pd.DataFrame,
        n: int = 10,
        sort_by: str = "Sharpe_Ratio",
    ) -> pd.DataFrame:
        """Return the top-N rows sorted by ``sort_by``."""
        if results_df.empty or sort_by not in results_df.columns:
            return results_df
        return (
            results_df[results_df["error"].isna()]
            .nlargest(n, sort_by)
            .reset_index(drop=True)
        )
