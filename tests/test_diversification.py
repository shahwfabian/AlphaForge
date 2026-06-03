"""Tests for src/analytics/diversification.py"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.diversification import (
    correlation_matrix,
    average_pairwise_correlation,
    most_correlated_pair,
    least_correlated_pair,
    portfolio_volatility,
    diversification_ratio,
    concentration_score,
    diversification_report,
)


@pytest.fixture
def multi_asset_returns():
    rng = np.random.default_rng(0)
    n   = 252
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    base = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({
        "A": base + rng.normal(0, 0.005, n),
        "B": base * 0.8 + rng.normal(0.0001, 0.008, n),
        "C": rng.normal(0.0002, 0.012, n),   # less correlated
    }, index=idx)


class TestCorrelationMatrix:
    def test_returns_dataframe(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        assert isinstance(corr, pd.DataFrame)

    def test_shape_n_by_n(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        n = multi_asset_returns.shape[1]
        assert corr.shape == (n, n)

    def test_diagonal_is_one(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)

    def test_symmetric(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)

    def test_values_bounded(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <=  1.0 + 1e-9).all()

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame()
        result = correlation_matrix(empty)
        assert result.empty

    def test_single_column_returns_empty(self):
        df = pd.DataFrame({"A": [0.01, 0.02]})
        result = correlation_matrix(df)
        assert result.empty


class TestAveragePairwiseCorrelation:
    def test_returns_float(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        avg  = average_pairwise_correlation(corr)
        assert isinstance(avg, float)

    def test_bounded(self, multi_asset_returns):
        corr = correlation_matrix(multi_asset_returns)
        avg  = average_pairwise_correlation(corr)
        assert -1.0 <= avg <= 1.0

    def test_single_asset_returns_nan(self):
        corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        assert np.isnan(average_pairwise_correlation(corr))

    def test_highly_correlated_close_to_one(self):
        # Two nearly identical series
        s = np.linspace(0, 1, 100)
        df = pd.DataFrame({"A": s, "B": s + 0.0001})
        corr = correlation_matrix(df)
        avg  = average_pairwise_correlation(corr)
        assert avg > 0.99


class TestMostAndLeastCorrelatedPair:
    def test_most_correlated_returns_tuple(self, multi_asset_returns):
        corr   = correlation_matrix(multi_asset_returns)
        result = most_correlated_pair(corr)
        assert isinstance(result, tuple) and len(result) == 3

    def test_most_correlated_value_within_range(self, multi_asset_returns):
        corr       = correlation_matrix(multi_asset_returns)
        _, _, val  = most_correlated_pair(corr)
        assert -1.0 <= val <= 1.0

    def test_least_correlated_returns_tuple(self, multi_asset_returns):
        corr   = correlation_matrix(multi_asset_returns)
        result = least_correlated_pair(corr)
        assert isinstance(result, tuple) and len(result) == 3

    def test_most_ge_least(self, multi_asset_returns):
        corr       = correlation_matrix(multi_asset_returns)
        _, _, most = most_correlated_pair(corr)
        _, _, least = least_correlated_pair(corr)
        assert most >= least

    def test_empty_returns_nans(self):
        empty  = pd.DataFrame()
        result = most_correlated_pair(empty)
        assert np.isnan(result[2])


class TestPortfolioVolatility:
    def test_returns_float(self, multi_asset_returns):
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        vol = portfolio_volatility(multi_asset_returns, weights)
        assert isinstance(vol, float)

    def test_positive(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.5}
        vol = portfolio_volatility(multi_asset_returns, weights)
        assert vol > 0

    def test_missing_tickers_returns_nan(self, multi_asset_returns):
        vol = portfolio_volatility(multi_asset_returns, {"XYZ": 1.0})
        assert np.isnan(vol)

    def test_single_asset_equals_individual_vol(self, multi_asset_returns):
        vol_port = portfolio_volatility(multi_asset_returns, {"A": 1.0})
        vol_indiv = float(multi_asset_returns["A"].std() * np.sqrt(252))
        assert abs(vol_port - vol_indiv) < 1e-6


class TestDiversificationRatio:
    def test_returns_float(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.5}
        dr = diversification_ratio(multi_asset_returns, weights)
        assert isinstance(dr, float)

    def test_greater_than_one_for_imperfect_correlation(self, multi_asset_returns):
        weights = {"A": 0.5, "C": 0.5}   # C is less correlated
        dr = diversification_ratio(multi_asset_returns, weights)
        assert dr >= 1.0

    def test_single_asset_returns_one(self, multi_asset_returns):
        dr = diversification_ratio(multi_asset_returns, {"A": 1.0})
        assert abs(dr - 1.0) < 1e-6


class TestConcentrationScore:
    def test_single_asset_is_one(self):
        assert concentration_score({"A": 1.0}) == pytest.approx(1.0)

    def test_equal_weight_is_one_over_n(self):
        n = 4
        weights = {str(i): 1/n for i in range(n)}
        hhi = concentration_score(weights)
        assert abs(hhi - 1/n) < 1e-9

    def test_empty_returns_nan(self):
        assert np.isnan(concentration_score({}))

    def test_more_assets_lower_score(self):
        h2 = concentration_score({"A": 0.5, "B": 0.5})
        h4 = concentration_score({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        assert h4 < h2


class TestDiversificationReport:
    def test_returns_dict(self, multi_asset_returns):
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        report  = diversification_report(multi_asset_returns, weights)
        assert isinstance(report, dict)

    def test_has_all_keys(self, multi_asset_returns):
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        report  = diversification_report(multi_asset_returns, weights)
        for key in ("corr_matrix", "avg_correlation", "portfolio_vol_ann",
                    "diversification_ratio", "concentration_score",
                    "most_correlated", "least_correlated"):
            assert key in report, f"Missing key: {key}"

    def test_corr_matrix_is_dataframe(self, multi_asset_returns):
        report = diversification_report(
            multi_asset_returns, {"A": 0.5, "B": 0.5}
        )
        assert isinstance(report["corr_matrix"], pd.DataFrame)
