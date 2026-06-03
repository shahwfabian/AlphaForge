"""
Tests for portfolio risk decomposition analytics.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk_decomposition import (
    rolling_beta_alpha,
    variance_decomposition,
    rolling_volatility_attribution,
    factor_sensitivity,
    sector_exposure,
    full_risk_decomposition,
    SECTOR_MAP,
)


@pytest.fixture
def aligned_returns():
    """Two correlated return series: strategy and benchmark."""
    rng = np.random.default_rng(42)
    n = 252
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    bm   = pd.Series(rng.normal(0.0004, 0.010, n), index=dates, name="bm")
    strat = bm * 0.85 + rng.normal(0.0002, 0.006, n)
    strat.name = "strat"
    return strat, bm


class TestRollingBetaAlpha:
    def test_returns_dataframe(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_beta_alpha(strat, bm, window=63)
        assert isinstance(result, pd.DataFrame)
        assert "Beta" in result.columns
        assert "Alpha" in result.columns
        assert "Alpha_Ann" in result.columns

    def test_length_matches_input(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_beta_alpha(strat, bm, window=63)
        assert len(result) == len(strat)

    def test_warm_up_is_nan(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_beta_alpha(strat, bm, window=63)
        # First 62 rows should be NaN
        assert result["Beta"].iloc[:62].isna().all()

    def test_beta_close_to_0point85(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_beta_alpha(strat, bm, window=63)
        stable_beta = result["Beta"].dropna()
        # Average rolling beta should be close to 0.85
        assert abs(stable_beta.mean() - 0.85) < 0.2

    def test_r_squared_between_0_and_1(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_beta_alpha(strat, bm, window=63)
        r2 = result["R_Squared"].dropna()
        assert (r2 >= 0).all()
        assert (r2 <= 1.0 + 1e-6).all()


class TestVarianceDecomposition:
    def test_returns_required_columns(self, aligned_returns):
        strat, bm = aligned_returns
        result = variance_decomposition(strat, bm, window=63)
        for col in ["Total_Variance", "Systematic_Variance",
                    "Idiosyncratic_Variance", "Systematic_Pct"]:
            assert col in result.columns

    def test_systematic_pct_bounded(self, aligned_returns):
        strat, bm = aligned_returns
        result = variance_decomposition(strat, bm, window=63)
        pct = result["Systematic_Pct"].dropna()
        assert (pct >= 0).all()
        assert (pct <= 1.0 + 1e-6).all()

    def test_idio_pct_plus_systematic_equals_one(self, aligned_returns):
        strat, bm = aligned_returns
        result = variance_decomposition(strat, bm, window=63)
        total = (result["Systematic_Pct"] + result["Idiosyncratic_Pct"]).dropna()
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


class TestVolatilityAttribution:
    def test_returns_dataframe(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_volatility_attribution(strat, bm, window=21)
        assert isinstance(result, pd.DataFrame)
        assert "Strategy_Vol" in result.columns
        assert "Benchmark_Vol" in result.columns

    def test_strategy_vol_positive(self, aligned_returns):
        strat, bm = aligned_returns
        result = rolling_volatility_attribution(strat, bm, window=21)
        vol = result["Strategy_Vol"].dropna()
        assert (vol >= 0).all()


class TestFactorSensitivity:
    def test_returns_dict(self, aligned_returns):
        strat, bm = aligned_returns
        result = factor_sensitivity(strat, bm, lookback_days=60)
        assert isinstance(result, dict)

    def test_required_keys(self, aligned_returns):
        strat, bm = aligned_returns
        result = factor_sensitivity(strat, bm, lookback_days=60)
        if result:  # May be empty if data is too short
            for key in ["alpha", "r_squared", "factor_loadings", "t_stats"]:
                assert key in result

    def test_r_squared_between_0_and_1(self, aligned_returns):
        strat, bm = aligned_returns
        result = factor_sensitivity(strat, bm, lookback_days=60)
        if result:
            assert 0 <= result["r_squared"] <= 1.0 + 1e-6

    def test_factor_loadings_has_three_factors(self, aligned_returns):
        strat, bm = aligned_returns
        result = factor_sensitivity(strat, bm, lookback_days=60)
        if result and "factor_loadings" in result:
            assert len(result["factor_loadings"]) == 3

    def test_insufficient_data_returns_empty(self):
        s = pd.Series(np.random.normal(0, 0.01, 10))
        b = pd.Series(np.random.normal(0, 0.01, 10))
        result = factor_sensitivity(s, b)
        assert result == {}


class TestSectorExposure:
    def test_known_tickers_mapped(self):
        result = sector_exposure(["AAPL", "MSFT", "JPM"])
        assert not result.empty
        assert "Sector" in result.columns
        assert "Sector_Weight" in result.columns

    def test_weights_sum_to_one(self):
        result = sector_exposure(["AAPL", "MSFT", "JPM"])
        assert abs(result["Sector_Weight"].sum() - 1.0) < 1e-6

    def test_equal_weights_default(self):
        tickers = ["AAPL", "MSFT"]
        result = sector_exposure(tickers)
        # Both are Technology → sector weight = 1.0
        tech_row = result[result["Sector"] == "Technology"]
        assert float(tech_row["Sector_Weight"].values[0]) == pytest.approx(1.0, abs=1e-6)

    def test_unknown_ticker_classified_as_unknown(self):
        result = sector_exposure(["XYZABC123"])
        assert "Unknown" in result["Sector"].values

    def test_empty_returns_empty_df(self):
        result = sector_exposure([])
        assert result.empty


class TestFullRiskDecomposition:
    def test_returns_all_keys(self, aligned_returns):
        strat, bm = aligned_returns
        result = full_risk_decomposition(strat, bm, "AAPL", window=63)
        expected_keys = [
            "rolling_beta_alpha", "variance_decomp",
            "vol_attribution", "factor_model", "sector_exposure"
        ]
        for key in expected_keys:
            assert key in result

    def test_sector_exposure_for_known_ticker(self, aligned_returns):
        strat, bm = aligned_returns
        result = full_risk_decomposition(strat, bm, "AAPL")
        se = result["sector_exposure"]
        assert not se.empty
        assert "Technology" in se["Sector"].values
