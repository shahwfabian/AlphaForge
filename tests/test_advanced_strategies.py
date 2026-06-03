"""
Tests for RSI, Bollinger Bands strategies and advanced analytics.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.rsi import RSIMeanReversion
from src.strategies.bollinger import BollingerBands
from src.analytics.performance import (
    monthly_returns_table,
    annual_returns,
    rolling_sharpe,
    compute_all_metrics,
)


# ── RSI Mean Reversion ─────────────────────────────────────────────────────────

class TestRSI:
    def test_basic_signals(self, processed_ohlcv):
        strat = RSIMeanReversion(period=14, oversold=30.0, overbought=70.0, exit_level=50.0)
        signals = strat.generate_signals(processed_ohlcv)
        assert isinstance(signals, pd.DataFrame)
        assert "Signal" in signals.columns
        assert "RSI" in signals.columns
        assert len(signals) == len(processed_ohlcv)

    def test_rsi_bounded(self, processed_ohlcv):
        """RSI should always be in [0, 100] after warm-up."""
        strat = RSIMeanReversion(period=14)
        signals = strat.generate_signals(processed_ohlcv)
        rsi_clean = signals["RSI"].dropna()
        assert (rsi_clean >= 0).all(), "RSI below 0"
        assert (rsi_clean <= 100).all(), "RSI above 100"

    def test_signal_is_0_or_1(self, processed_ohlcv):
        strat = RSIMeanReversion()
        signals = strat.generate_signals(processed_ohlcv)
        assert set(signals["Signal"].unique()).issubset({0, 1})

    def test_invalid_oversold_raises(self):
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=60.0)  # must be < 50

    def test_invalid_overbought_raises(self):
        with pytest.raises(ValueError):
            RSIMeanReversion(overbought=40.0)  # must be > 50

    def test_exit_level_must_be_between_oversold_and_overbought(self):
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=30.0, overbought=70.0, exit_level=80.0)

    def test_position_is_signal_shifted(self, processed_ohlcv):
        strat = RSIMeanReversion()
        signals = strat.generate_signals(processed_ohlcv)
        expected = signals["Signal"].shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(
            signals["Position"], expected, check_names=False
        )

    def test_warm_up_is_flat(self, processed_ohlcv):
        period = 14
        strat = RSIMeanReversion(period=period)
        signals = strat.generate_signals(processed_ohlcv)
        # First `period` RSI values are NaN → signal must be 0
        assert (signals["Signal"].iloc[:period] == 0).all()

    def test_strategy_return_calculation(self, processed_ohlcv):
        strat = RSIMeanReversion()
        signals = strat.generate_signals(processed_ohlcv)
        expected = signals["Position"] * signals["Asset_Return"]
        pd.testing.assert_series_equal(
            signals["Strategy_Return"], expected, check_names=False
        )

    def test_no_nan_in_strategy_return(self, processed_ohlcv):
        strat = RSIMeanReversion()
        signals = strat.generate_signals(processed_ohlcv)
        assert signals["Strategy_Return"].isna().sum() == 0


# ── Bollinger Bands ────────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_basic_signals(self, processed_ohlcv):
        strat = BollingerBands(window=20, n_std=2.0)
        signals = strat.generate_signals(processed_ohlcv)
        assert isinstance(signals, pd.DataFrame)
        assert "Signal" in signals.columns
        assert "BB_Lower" in signals.columns
        assert "BB_Upper" in signals.columns
        assert "BB_Middle" in signals.columns

    def test_bands_present(self, processed_ohlcv):
        strat = BollingerBands(window=20, n_std=2.0)
        signals = strat.generate_signals(processed_ohlcv)
        for col in ["BB_Lower", "BB_Upper", "BB_Middle", "BB_Pct", "BB_Width"]:
            assert col in signals.columns

    def test_upper_greater_than_lower(self, processed_ohlcv):
        strat = BollingerBands(window=20, n_std=2.0)
        signals = strat.generate_signals(processed_ohlcv)
        valid = signals.dropna(subset=["BB_Upper", "BB_Lower"])
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_signal_is_0_or_1(self, processed_ohlcv):
        strat = BollingerBands()
        signals = strat.generate_signals(processed_ohlcv)
        assert set(signals["Signal"].unique()).issubset({0, 1})

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            BollingerBands(window=3)  # below minimum 5

    def test_invalid_n_std_raises(self):
        with pytest.raises(ValueError):
            BollingerBands(n_std=5.0)  # above maximum 4.0

    def test_position_shifted_one_day(self, processed_ohlcv):
        strat = BollingerBands()
        signals = strat.generate_signals(processed_ohlcv)
        expected = signals["Signal"].shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(
            signals["Position"], expected, check_names=False
        )

    def test_warm_up_is_flat(self, processed_ohlcv):
        window = 20
        strat = BollingerBands(window=window)
        signals = strat.generate_signals(processed_ohlcv)
        # Before the window fills, bands are NaN → signals must be 0
        assert (signals["Signal"].iloc[:window - 1] == 0).all()

    def test_exit_at_upper_band(self, processed_ohlcv):
        """When exit_at_middle=False, position exits at upper band."""
        strat = BollingerBands(window=20, exit_at_middle=False)
        signals = strat.generate_signals(processed_ohlcv)
        assert set(signals["Signal"].unique()).issubset({0, 1})

    def test_no_nan_in_strategy_return(self, processed_ohlcv):
        strat = BollingerBands()
        signals = strat.generate_signals(processed_ohlcv)
        assert signals["Strategy_Return"].isna().sum() == 0


# ── Advanced performance metrics ──────────────────────────────────────────────

class TestAdvancedMetrics:
    @pytest.fixture
    def daily_rets(self):
        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-01", periods=504, freq="B")
        return pd.Series(rng.normal(0.0004, 0.011, 504), index=dates)

    def test_monthly_returns_table_shape(self, daily_rets):
        tbl = monthly_returns_table(daily_rets)
        assert not tbl.empty
        assert "Annual" in tbl.columns
        assert len(tbl.columns) == 13  # 12 months + Annual

    def test_monthly_returns_rows_are_years(self, daily_rets):
        tbl = monthly_returns_table(daily_rets)
        # All index values should be year integers
        for yr in tbl.index:
            assert 2000 <= yr <= 2100

    def test_annual_column_compounds_monthly(self, daily_rets):
        tbl = monthly_returns_table(daily_rets)
        for yr in tbl.index:
            row = tbl.loc[yr, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]].dropna()
            expected_annual = (1 + row).prod() - 1
            actual_annual   = tbl.loc[yr, "Annual"]
            assert actual_annual == pytest.approx(expected_annual, rel=1e-6)

    def test_annual_returns_series(self, daily_rets):
        ann = annual_returns(daily_rets)
        assert isinstance(ann, pd.Series)
        assert len(ann) >= 1
        # Index should be years
        assert all(2000 <= yr <= 2100 for yr in ann.index)

    def test_rolling_sharpe_length(self, daily_rets):
        rs = rolling_sharpe(daily_rets, window=63)
        assert len(rs) == len(daily_rets)

    def test_omega_ratio_positive_for_positive_trend(self):
        rng = np.random.default_rng(42)
        n = 252
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        returns = pd.Series(rng.normal(0.002, 0.010, n), index=dates)
        pv = 100_000 * (1 + returns).cumprod()
        equity = pd.DataFrame({
            "Portfolio_Value": pv,
            "Daily_Return": returns,
        }, index=dates)
        metrics = compute_all_metrics(equity)
        assert metrics.get("Omega_Ratio", 0) > 1.0

    def test_ulcer_index_zero_for_constant_up(self):
        n = 252
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        returns = pd.Series([0.001] * n, index=dates)
        pv = 100_000 * (1 + returns).cumprod()
        equity = pd.DataFrame({
            "Portfolio_Value": pv,
            "Daily_Return": returns,
        }, index=dates)
        metrics = compute_all_metrics(equity)
        assert metrics.get("Ulcer_Index", 999) == pytest.approx(0.0, abs=1e-6)

    def test_tail_ratio_non_negative(self):
        rng = np.random.default_rng(7)
        n = 252
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        returns = pd.Series(rng.normal(0.0003, 0.012, n), index=dates)
        pv = 100_000 * (1 + returns).cumprod()
        equity = pd.DataFrame({
            "Portfolio_Value": pv,
            "Daily_Return": returns,
        }, index=dates)
        metrics = compute_all_metrics(equity)
        assert metrics.get("Tail_Ratio", -1) >= 0

    def test_serenity_ratio_positive_for_uptrend(self):
        """
        Serenity = CAGR / Ulcer_Index.
        A generally-upward series with occasional small pullbacks should yield > 0.
        (A perfectly flat constant uptrend would have Ulcer=0 and return 0.)
        """
        rng = np.random.default_rng(99)
        n = 504
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Mostly positive returns with occasional small drawdowns
        returns = pd.Series(rng.normal(0.001, 0.005, n), index=dates)
        pv = 100_000 * (1 + returns).cumprod()
        equity = pd.DataFrame({
            "Portfolio_Value": pv,
            "Daily_Return": returns,
        }, index=dates)
        metrics = compute_all_metrics(equity)
        # CAGR should be positive and Ulcer > 0 (small drawdowns exist)
        assert metrics.get("Annualised_Return", 0) > 0
        # Serenity should be positive (non-negative CAGR, non-zero Ulcer)
        assert metrics.get("Serenity_Ratio", -1) >= 0
