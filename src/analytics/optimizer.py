"""
Strategy parameter optimisation — grid search and randomized search.

WARNING: Optimising strategy parameters on historical data is prone to
overfitting.  In-sample Sharpe will almost always exceed out-of-sample
Sharpe.  Always pair with walk-forward validation before drawing
any conclusions.
"""

import itertools
import logging
import random
from typing import Any

import numpy as np
import pandas as pd

from ..analytics.performance import compute_all_metrics

logger = logging.getLogger(__name__)


# ── Generic optimiser class ───────────────────────────────────────────────────

class StrategyOptimizer:
    """
    Flexible strategy parameter optimiser.

    Supports:
    - **Exhaustive grid search** — test every combination in the Cartesian product.
    - **Randomized search** — randomly sample from the grid (faster for large spaces).
    - **Successive halving** — iteratively prune worst-performing configurations.

    Parameters
    ----------
    strategy_name : str
        Registered strategy name (must exist in ``src.strategies.registry``).
    param_grid : dict[str, list]
        Mapping of parameter name → candidate values.
    initial_capital : float
    transaction_cost : float
    slippage : float
    """

    def __init__(
        self,
        strategy_name: str,
        param_grid: dict[str, list[Any]],
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.strategy_name    = strategy_name
        self.param_grid       = param_grid
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage         = slippage

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        df: pd.DataFrame,
        ticker: str = "ASSET",
        sort_by: str = "Sharpe_Ratio",
    ) -> pd.DataFrame:
        """
        Exhaustive grid search over all parameter combinations.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed OHLCV data (full in-sample period).
        ticker : str
        sort_by : str
            Metric to rank results by.

        Returns
        -------
        pd.DataFrame
            One row per valid combination, sorted by ``sort_by`` descending.
        """
        combos = self._all_combos()
        logger.info(
            "Grid search: %s | %d combinations.", self.strategy_name, len(combos)
        )
        return self._run_combos(df, ticker, combos, sort_by)

    # ------------------------------------------------------------------
    # Randomized search
    # ------------------------------------------------------------------

    def randomized_search(
        self,
        df: pd.DataFrame,
        n_iter: int = 50,
        ticker: str = "ASSET",
        sort_by: str = "Sharpe_Ratio",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Randomly sample ``n_iter`` parameter combinations.

        More efficient than exhaustive grid search when the parameter space
        is large.  For large grids, randomized search often finds near-optimal
        parameters in far fewer evaluations (Bergstra & Bengio, 2012).

        Parameters
        ----------
        df : pd.DataFrame
        n_iter : int
            Number of random combinations to evaluate.
        ticker : str
        sort_by : str
        random_seed : int

        Returns
        -------
        pd.DataFrame
        """
        all_combos = self._all_combos()
        rng = random.Random(random_seed)
        if n_iter < len(all_combos):
            combos = rng.sample(all_combos, n_iter)
        else:
            combos = all_combos

        logger.info(
            "Randomized search: %s | %d / %d combinations.",
            self.strategy_name, len(combos), len(all_combos),
        )
        return self._run_combos(df, ticker, combos, sort_by)

    # ------------------------------------------------------------------
    # Successive halving
    # ------------------------------------------------------------------

    def successive_halving(
        self,
        df: pd.DataFrame,
        ticker: str = "ASSET",
        n_initial: int = 64,
        halving_factor: int = 2,
        sort_by: str = "Sharpe_Ratio",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Successive halving (SHA) optimisation.

        Start with ``n_initial`` random configurations, evaluate on a subset
        of data, keep the top half, repeat with more data.
        Terminates when one configuration remains or data is exhausted.

        Based on: Jamieson & Talwalkar (2016). "Non-stochastic best arm
        identification and hyperparameter optimization."

        Parameters
        ----------
        df : pd.DataFrame
        ticker : str
        n_initial : int
            Starting number of random configurations.
        halving_factor : int
            Fraction of survivors kept at each round.
        sort_by : str
        random_seed : int

        Returns
        -------
        pd.DataFrame
            Remaining configurations after all halving rounds.
        """
        all_combos = self._all_combos()
        rng = random.Random(random_seed)
        candidates = rng.sample(
            all_combos, min(n_initial, len(all_combos))
        )

        round_num = 0
        n_rounds  = int(np.ceil(np.log2(max(len(candidates), 1))))
        total_rows = len(df)

        logger.info(
            "Successive halving: %s | %d initial | %d rounds.",
            self.strategy_name, len(candidates), n_rounds,
        )

        while len(candidates) > 1 and round_num < n_rounds:
            # Use progressively more data each round
            frac = min(1.0, (round_num + 1) / n_rounds)
            n_rows = max(60, int(total_rows * frac))
            sub_df = df.iloc[:n_rows]

            results = self._run_combos(sub_df, ticker, candidates, sort_by)
            results = results[results["error"].isna() if "error" in results.columns else results.index.notna()]

            if results.empty:
                break

            keep = max(1, len(results) // halving_factor)
            top_params = results.head(keep)["params"].tolist()
            candidates = [c for c in candidates if c in top_params]

            round_num += 1
            logger.info(
                "SHA round %d: evaluated %d | kept %d | data fraction %.0f%%.",
                round_num, len(results), keep, frac * 100,
            )

        return self._run_combos(df, ticker, candidates, sort_by)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_combos(self) -> list[dict[str, Any]]:
        """Return all Cartesian product combinations of param_grid."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _run_combos(
        self,
        df: pd.DataFrame,
        ticker: str,
        combos: list[dict[str, Any]],
        sort_by: str,
    ) -> pd.DataFrame:
        """Evaluate a list of parameter dicts and return a results DataFrame."""
        from ..strategies.registry import build_strategy
        from ..backtester.engine import BacktestEngine

        rows = []
        for params in combos:
            try:
                strategy = build_strategy(self.strategy_name, params)
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    slippage=self.slippage,
                )
                result  = engine.run(df, ticker)
                metrics = compute_all_metrics(result["equity_curve"])
                rows.append({
                    "params":    params,
                    "error":     None,
                    **params,           # flatten params as columns
                    **metrics,
                })
            except Exception as e:
                logger.debug(
                    "Skipping %s params=%s: %s", self.strategy_name, params, e
                )
                rows.append({
                    "params": params,
                    "error":  str(e),
                    **params,
                })

        if not rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(rows)
        valid = result_df[result_df.get("error", pd.Series(dtype=str)).isna()].copy()
        if sort_by in valid.columns:
            valid.sort_values(sort_by, ascending=False, inplace=True)
        valid.reset_index(drop=True, inplace=True)
        return valid


# ── Convenience wrappers (backward compatible) ────────────────────────────────

def grid_search_ma(
    df: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Grid search over Moving Average Crossover parameters."""
    opt = StrategyOptimizer(
        "Moving Average Crossover",
        {"short_window": short_windows, "long_window": long_windows},
        initial_capital, transaction_cost, slippage,
    )
    return opt.grid_search(df)


def grid_search_mr(
    df: pd.DataFrame,
    windows: list[int],
    entry_thresholds: list[float],
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Grid search over Mean Reversion parameters."""
    exit_thresholds = [round(e * 0.25, 2) for e in entry_thresholds]
    valid_pairs = [
        (w, e, x)
        for w in windows
        for e, x in zip(entry_thresholds, exit_thresholds)
        if x < e
    ]
    param_dicts = [
        {"window": w, "entry_threshold": e, "exit_threshold": x}
        for w, e, x in valid_pairs
    ]
    opt = StrategyOptimizer(
        "Mean Reversion", {}, initial_capital, transaction_cost, slippage
    )
    return opt._run_combos(df, "ASSET", param_dicts, "Sharpe_Ratio")


def randomized_search(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    n_iter: int = 50,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Randomized search over any strategy's parameter grid."""
    opt = StrategyOptimizer(
        strategy_name, param_grid, initial_capital, transaction_cost, slippage
    )
    return opt.randomized_search(df, n_iter=n_iter, random_seed=random_seed)
