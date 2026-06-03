"""
Tests for the parallel backtesting engine and strategy registry.
Uses ThreadPoolExecutor (n_workers=2) for speed and CI compatibility.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester.parallel_engine import ParallelBacktester, _worker
from src.strategies.registry import (
    build_strategy,
    get_strategy_class,
    list_strategies,
    register,
    discover_strategies,
    strategy_info,
)
from src.strategies.base_strategy import BaseStrategy


# ── Worker function tests ─────────────────────────────────────────────────────

class TestWorkerFunction:
    def test_worker_returns_dict(self, processed_ohlcv):
        task = {
            "df":               processed_ohlcv,
            "ticker":           "TEST",
            "strategy_name":    "Buy & Hold",
            "params":           {},
            "initial_capital":  100_000.0,
            "transaction_cost": 0.001,
            "slippage":         0.0005,
            "vol_target":       None,
        }
        result = _worker(task)
        assert isinstance(result, dict)
        assert result.get("error") is None

    def test_worker_returns_metrics(self, processed_ohlcv):
        task = {
            "df": processed_ohlcv, "ticker": "TEST",
            "strategy_name": "Moving Average Crossover",
            "params": {"short_window": 10, "long_window": 30},
            "initial_capital": 100_000.0,
            "transaction_cost": 0.001,
            "slippage": 0.0005,
            "vol_target": None,
        }
        result = _worker(task)
        assert "Sharpe_Ratio" in result
        assert "Cumulative_Return" in result

    def test_worker_captures_errors_gracefully(self, processed_ohlcv):
        task = {
            "df": processed_ohlcv, "ticker": "TEST",
            "strategy_name": "Moving Average Crossover",
            "params": {"short_window": 50, "long_window": 10},  # invalid (short > long)
            "initial_capital": 100_000.0,
            "transaction_cost": 0.001,
            "slippage": 0.0005,
            "vol_target": None,
        }
        result = _worker(task)
        assert result.get("error") is not None  # should be captured, not raised


# ── Parallel backtester tests ─────────────────────────────────────────────────

class TestParallelBacktester:
    @pytest.fixture
    def backtester(self):
        return ParallelBacktester(
            initial_capital=100_000.0,
            transaction_cost=0.001,
            slippage=0.0005,
            executor="thread",
            n_workers=2,
        )

    def test_parameter_sweep_returns_dataframe(self, backtester, processed_ohlcv):
        results = backtester.run_parameter_sweep(
            df=processed_ohlcv,
            ticker="TEST",
            strategy_name="Moving Average Crossover",
            param_grid={"short_window": [10, 20], "long_window": [30, 50]},
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1  # At least some valid combos

    def test_parameter_sweep_sorted_by_sharpe(self, backtester, processed_ohlcv):
        results = backtester.run_parameter_sweep(
            df=processed_ohlcv,
            ticker="TEST",
            strategy_name="Moving Average Crossover",
            param_grid={"short_window": [10, 20], "long_window": [40, 60]},
        )
        if len(results) > 1 and "Sharpe_Ratio" in results.columns:
            assert results["Sharpe_Ratio"].iloc[0] >= results["Sharpe_Ratio"].iloc[-1]

    def test_randomized_sweep(self, backtester, processed_ohlcv):
        results = backtester.run_parameter_sweep(
            df=processed_ohlcv,
            ticker="TEST",
            strategy_name="Moving Average Crossover",
            param_grid={
                "short_window": [5, 10, 15, 20, 25],
                "long_window":  [30, 40, 50, 60, 80],
            },
            randomized=True,
            max_combinations=4,
        )
        assert len(results) <= 4

    def test_multi_asset_sweep(self, backtester, processed_ohlcv):
        datasets = {
            "ASSET_A": processed_ohlcv,
            "ASSET_B": processed_ohlcv.copy(),
        }
        results = backtester.run_multi_asset(
            datasets=datasets,
            strategy_name="Buy & Hold",
            params={},
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1

    def test_strategy_sweep(self, backtester, processed_ohlcv):
        strategies = [
            ("Buy & Hold",               {}),
            ("Moving Average Crossover", {"short_window": 10, "long_window": 30}),
            ("Bollinger Bands",          {"window": 20, "n_std": 2.0}),
        ]
        results = backtester.run_strategy_sweep(
            df=processed_ohlcv,
            ticker="TEST",
            strategies=strategies,
        )
        assert isinstance(results, pd.DataFrame)

    def test_top_n_filters_errors(self, backtester, processed_ohlcv):
        results = backtester.run_parameter_sweep(
            df=processed_ohlcv,
            ticker="TEST",
            strategy_name="Moving Average Crossover",
            param_grid={"short_window": [10], "long_window": [30]},
        )
        top = ParallelBacktester.top_n(results, n=5)
        assert isinstance(top, pd.DataFrame)


# ── Strategy registry tests ───────────────────────────────────────────────────

class TestStrategyRegistry:
    def test_all_builtin_strategies_registered(self):
        registered = list_strategies()
        expected = [
            "Buy & Hold",
            "Bollinger Bands",
            "Mean Reversion",
            "Momentum",
            "Moving Average Crossover",
            "RSI Mean Reversion",
        ]
        for name in expected:
            assert name in registered, f"'{name}' not in registry"

    def test_build_strategy_returns_correct_type(self):
        from src.strategies.moving_average import MovingAverageCrossover
        s = build_strategy("Moving Average Crossover", {"short_window": 10, "long_window": 30})
        assert isinstance(s, MovingAverageCrossover)

    def test_build_strategy_with_empty_params(self):
        s = build_strategy("Buy & Hold", {})
        assert s is not None

    def test_get_strategy_class(self):
        cls = get_strategy_class("Momentum")
        from src.strategies.momentum import Momentum
        assert cls is Momentum

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_strategy_class("NonExistentStrategy")

    def test_build_strategy_unknown_raises(self):
        with pytest.raises(KeyError):
            build_strategy("Ghost Strategy", {})

    def test_register_custom_strategy(self):
        class DummyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Dummy", {})
            def validate_parameters(self): pass
            def generate_signals(self, df): return self._base_output(df)

        register(DummyStrategy, name="Dummy Test Strategy")
        assert "Dummy Test Strategy" in list_strategies()

        s = build_strategy("Dummy Test Strategy", {})
        assert isinstance(s, DummyStrategy)

    def test_strategy_info_returns_dict(self):
        info = strategy_info("Bollinger Bands")
        assert "name" in info
        assert "class_name" in info
        assert info["name"] == "Bollinger Bands"

    def test_discover_strategies_does_not_crash(self):
        """discover_strategies should run without error even if all are already registered."""
        count = discover_strategies("src.strategies")
        assert count >= 0  # May be 0 if all already registered
