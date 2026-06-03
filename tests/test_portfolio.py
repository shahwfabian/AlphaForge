"""Tests for src/portfolio/allocation.py"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.allocation import (
    equal_weight_allocation,
    normalize_weights,
    validate_weights,
    calculate_portfolio_returns,
    allocation_summary,
    dominant_asset_warning,
    cumulative_portfolio_value,
)


class TestEqualWeightAllocation:
    def test_returns_dict(self):
        result = equal_weight_allocation(["AAPL", "MSFT"])
        assert isinstance(result, dict)

    def test_correct_number_of_keys(self):
        result = equal_weight_allocation(["A", "B", "C"])
        assert len(result) == 3

    def test_weights_sum_to_one(self):
        result = equal_weight_allocation(["A", "B", "C", "D"])
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_returns_empty(self):
        assert equal_weight_allocation([]) == {}

    def test_single_asset_weight_is_one(self):
        result = equal_weight_allocation(["SPY"])
        assert abs(result["SPY"] - 1.0) < 1e-9


class TestNormalizeWeights:
    def test_sums_to_one(self):
        raw = {"A": 30.0, "B": 50.0, "C": 20.0}
        norm = normalize_weights(raw)
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_already_normalised_unchanged(self):
        raw = {"A": 0.5, "B": 0.5}
        norm = normalize_weights(raw)
        assert abs(norm["A"] - 0.5) < 1e-9

    def test_negatives_clamped_to_zero(self):
        raw = {"A": -5.0, "B": 10.0}
        norm = normalize_weights(raw)
        assert norm["A"] == 0.0
        assert norm["B"] == pytest.approx(1.0)

    def test_all_zeros_falls_back_to_equal(self):
        raw = {"A": 0.0, "B": 0.0}
        norm = normalize_weights(raw)
        assert abs(norm["A"] - 0.5) < 1e-9

    def test_empty_returns_empty(self):
        assert normalize_weights({}) == {}


class TestValidateWeights:
    def test_valid_weights(self):
        ok, msg = validate_weights({"A": 0.6, "B": 0.4})
        assert ok is True

    def test_empty_is_invalid(self):
        ok, _ = validate_weights({})
        assert ok is False

    def test_negative_weight_invalid(self):
        ok, msg = validate_weights({"A": -0.1, "B": 1.1})
        assert ok is False
        assert "negative" in msg.lower()

    def test_sum_not_one_invalid(self):
        ok, msg = validate_weights({"A": 0.3, "B": 0.3})
        assert ok is False
        assert "sum" in msg.lower()

    def test_tolerance_accepted(self):
        # 0.990 sum is within 1.5% tolerance → should pass
        ok, _ = validate_weights({"A": 0.495, "B": 0.495})
        assert ok is True

    def test_exact_sum_accepted(self):
        ok, _ = validate_weights({"A": 0.5, "B": 0.5})
        assert ok is True


class TestCalculatePortfolioReturns:
    @pytest.fixture
    def simple_returns(self):
        n   = 50
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.01, n),
            "MSFT": np.random.normal(0.001, 0.01, n),
            "SPY":  np.random.normal(0.0005, 0.008, n),
        }, index=idx)

    def test_returns_series(self, simple_returns):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result  = calculate_portfolio_returns(simple_returns, weights)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, simple_returns):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result  = calculate_portfolio_returns(simple_returns, weights)
        assert len(result) == len(simple_returns)

    def test_equal_weight_average(self, simple_returns):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result  = calculate_portfolio_returns(simple_returns, weights)
        expected = 0.5 * simple_returns["AAPL"] + 0.5 * simple_returns["MSFT"]
        np.testing.assert_allclose(result.values, expected.values, atol=1e-10)

    def test_no_matching_tickers_raises(self, simple_returns):
        with pytest.raises(ValueError, match="None of the weight tickers"):
            calculate_portfolio_returns(simple_returns, {"XYZABC": 1.0})

    def test_partial_match_normalises(self, simple_returns):
        weights = {"AAPL": 0.5, "XYZABC": 0.5}   # XYZABC missing → 100% AAPL
        result  = calculate_portfolio_returns(simple_returns, weights)
        np.testing.assert_allclose(result.values, simple_returns["AAPL"].values, atol=1e-10)

    def test_named_portfolio(self, simple_returns):
        result = calculate_portfolio_returns(simple_returns, {"AAPL": 1.0})
        assert result.name == "Portfolio"


class TestAllocationSummary:
    def test_returns_dataframe(self):
        df = allocation_summary({"A": 0.6, "B": 0.4})
        assert isinstance(df, pd.DataFrame)

    def test_sorted_largest_first(self):
        df = allocation_summary({"A": 0.3, "B": 0.7})
        assert df["Symbol"].iloc[0] == "B"

    def test_empty_returns_empty_df(self):
        df = allocation_summary({})
        assert df.empty

    def test_weight_pct_sums_to_100(self):
        df = allocation_summary({"A": 50, "B": 30, "C": 20})
        assert abs(df["Weight_Pct"].sum() - 100.0) < 0.01


class TestDominantAssetWarning:
    def test_no_warning_for_balanced(self):
        assert dominant_asset_warning({"A": 0.5, "B": 0.5}) is None

    def test_warning_when_dominant(self):
        msg = dominant_asset_warning({"A": 0.9, "B": 0.1})
        assert msg is not None
        assert "A" in msg

    def test_empty_no_warning(self):
        assert dominant_asset_warning({}) is None


class TestCumulativePortfolioValue:
    def test_starts_at_initial_value(self):
        rets = pd.Series([0.01, 0.02, -0.01])
        pv   = cumulative_portfolio_value(rets, 100_000.0)
        assert abs(pv.iloc[0] - 101_000.0) < 1.0

    def test_length_matches_returns(self):
        rets = pd.Series([0.01] * 20)
        pv   = cumulative_portfolio_value(rets, 50_000.0)
        assert len(pv) == 20
