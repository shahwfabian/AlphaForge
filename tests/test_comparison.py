"""Tests for src/analytics/comparison.py"""

import numpy as np
import pytest
import pandas as pd

from src.analytics.comparison import (
    build_comparison_table,
    rank_strategies,
    format_comparison_table,
    best_strategy,
    COMPARISON_METRICS,
)

SAMPLE_METRICS = {
    "Strategy A": {
        "Sharpe_Ratio": 1.2, "CAGR": 0.15, "Max_Drawdown": -0.12,
        "Calmar_Ratio": 1.25, "Win_Rate": 0.55, "Profit_Factor": 1.5,
        "Sortino_Ratio": 1.8, "Volatility_Ann": 0.14, "VaR_95_Historical": -0.015,
        "Cumulative_Return": 0.45,
    },
    "Strategy B": {
        "Sharpe_Ratio": 0.8, "CAGR": 0.10, "Max_Drawdown": -0.20,
        "Calmar_Ratio": 0.50, "Win_Rate": 0.48, "Profit_Factor": 1.2,
        "Sortino_Ratio": 1.1, "Volatility_Ann": 0.18, "VaR_95_Historical": -0.021,
        "Cumulative_Return": 0.25,
    },
    "Buy & Hold": {
        "Sharpe_Ratio": 0.9, "CAGR": 0.11, "Max_Drawdown": -0.15,
        "Calmar_Ratio": 0.73, "Win_Rate": 0.53, "Profit_Factor": 1.3,
        "Sortino_Ratio": 1.2, "Volatility_Ann": 0.16, "VaR_95_Historical": -0.018,
        "Cumulative_Return": 0.30,
    },
}


class TestBuildComparisonTable:
    def test_returns_dataframe(self):
        df = build_comparison_table(SAMPLE_METRICS)
        assert isinstance(df, pd.DataFrame)

    def test_index_is_strategy_names(self):
        df = build_comparison_table(SAMPLE_METRICS)
        assert set(df.index) == set(SAMPLE_METRICS.keys())

    def test_columns_are_comparison_metrics(self):
        df = build_comparison_table(SAMPLE_METRICS)
        for col in COMPARISON_METRICS:
            assert col in df.columns

    def test_values_preserved(self):
        df = build_comparison_table(SAMPLE_METRICS)
        assert df.loc["Strategy A", "Sharpe_Ratio"] == pytest.approx(1.2)

    def test_empty_input_empty_df(self):
        df = build_comparison_table({})
        assert df.empty


class TestRankStrategies:
    def test_returns_dataframe(self):
        df = rank_strategies(SAMPLE_METRICS)
        assert isinstance(df, pd.DataFrame)

    def test_has_rank_column(self):
        df = rank_strategies(SAMPLE_METRICS)
        assert "Rank" in df.columns

    def test_rank_starts_at_one(self):
        df = rank_strategies(SAMPLE_METRICS)
        assert df["Rank"].iloc[0] == 1

    def test_default_sort_by_sharpe(self):
        df = rank_strategies(SAMPLE_METRICS)
        # Strategy A has highest Sharpe (1.2), should be first
        assert df.index[0] == "Strategy A"

    def test_sort_by_max_drawdown_descending(self):
        # Max_Drawdown values are negative; higher (less negative) = smaller loss = better
        # Sort descending → -0.12 > -0.15 > -0.20, so Strategy A ranks first
        df = rank_strategies(SAMPLE_METRICS, by="Max_Drawdown")
        assert df.index[0] == "Strategy A"

    def test_sort_by_cagr(self):
        df = rank_strategies(SAMPLE_METRICS, by="CAGR")
        assert df.index[0] == "Strategy A"


class TestFormatComparisonTable:
    def test_returns_dataframe(self):
        raw = build_comparison_table(SAMPLE_METRICS)
        fmt = format_comparison_table(raw)
        assert isinstance(fmt, pd.DataFrame)

    def test_pct_columns_are_strings(self):
        raw = rank_strategies(SAMPLE_METRICS)
        fmt = format_comparison_table(raw)
        if "CAGR" in fmt.columns:
            # CAGR values should be formatted strings like "+15.00%"
            val = fmt["CAGR"].iloc[0]
            assert isinstance(val, str)
            assert "%" in val

    def test_ratio_columns_are_strings(self):
        raw = rank_strategies(SAMPLE_METRICS)
        fmt = format_comparison_table(raw)
        if "Sharpe_Ratio" in fmt.columns:
            val = fmt["Sharpe_Ratio"].iloc[0]
            assert isinstance(val, str)


class TestBestStrategy:
    def test_returns_string(self):
        result = best_strategy(SAMPLE_METRICS)
        assert isinstance(result, str)

    def test_returns_highest_sharpe(self):
        result = best_strategy(SAMPLE_METRICS, by="Sharpe_Ratio")
        assert result == "Strategy A"

    def test_empty_returns_none(self):
        assert best_strategy({}) is None

    def test_sort_by_calmar(self):
        result = best_strategy(SAMPLE_METRICS, by="Calmar_Ratio")
        assert result == "Strategy A"
