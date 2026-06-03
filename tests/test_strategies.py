"""
Tests for all trading strategy implementations.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.moving_average import MovingAverageCrossover
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.buy_hold import BuyAndHold


# ── Shared signal-quality checks ───────────────────────────────────────────────

def _assert_valid_output(signals_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """Common assertions for any strategy output DataFrame."""
    assert isinstance(signals_df, pd.DataFrame)
    assert len(signals_df) == len(df)
    assert "Signal" in signals_df.columns
    assert "Position" in signals_df.columns
    assert "Strategy_Return" in signals_df.columns
    assert "Asset_Return" in signals_df.columns
    # Signals should be 0 or 1 (long-only strategies)
    assert set(signals_df["Signal"].unique()).issubset({0, 1})
    # No NaN in Strategy_Return
    assert signals_df["Strategy_Return"].isna().sum() == 0


# ── Moving Average Crossover ───────────────────────────────────────────────────

class TestMovingAverageCrossover:
    def test_basic_signals(self, processed_ohlcv):
        """Strategy should produce valid signals on normal data."""
        strat = MovingAverageCrossover(short_window=10, long_window=30)
        signals = strat.generate_signals(processed_ohlcv)
        _assert_valid_output(signals, processed_ohlcv)

    def test_warm_up_period_is_flat(self, processed_ohlcv):
        """Signals during the warm-up period should be 0 (no long_window yet)."""
        strat = MovingAverageCrossover(short_window=10, long_window=30)
        signals = strat.generate_signals(processed_ohlcv)
        # First 29 rows (long_window-1) should have no long signal
        warm_up = signals["Signal"].iloc[:29]
        assert (warm_up == 0).all(), "Warm-up period should be flat (no signal)"

    def test_short_must_be_less_than_long(self):
        """Constructor should raise if short_window >= long_window."""
        with pytest.raises(ValueError):
            MovingAverageCrossover(short_window=50, long_window=20)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            MovingAverageCrossover(short_window=1, long_window=50)

    def test_position_shifted_one_day(self, processed_ohlcv):
        """Position should be Signal shifted by 1 day (no lookahead)."""
        strat = MovingAverageCrossover(short_window=10, long_window=30)
        signals = strat.generate_signals(processed_ohlcv)
        expected_position = signals["Signal"].shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(
            signals["Position"], expected_position, check_names=False
        )

    def test_strategy_return_is_position_times_asset_return(self, processed_ohlcv):
        """Strategy_Return = Position * Asset_Return by definition."""
        strat = MovingAverageCrossover(short_window=10, long_window=30)
        signals = strat.generate_signals(processed_ohlcv)
        expected = signals["Position"] * signals["Asset_Return"]
        pd.testing.assert_series_equal(
            signals["Strategy_Return"], expected, check_names=False
        )


# ── Mean Reversion ─────────────────────────────────────────────────────────────

class TestMeanReversion:
    def test_basic_signals(self, processed_ohlcv):
        strat = MeanReversion(window=20, entry_threshold=2.0, exit_threshold=0.5)
        signals = strat.generate_signals(processed_ohlcv)
        _assert_valid_output(signals, processed_ohlcv)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            MeanReversion(window=4)  # Below minimum

        with pytest.raises(ValueError):
            MeanReversion(window=20, entry_threshold=-1.0)  # Negative threshold

        with pytest.raises(ValueError):
            MeanReversion(window=20, entry_threshold=1.0, exit_threshold=1.5)  # exit >= entry

    def test_entry_after_low_zscore(self, processed_ohlcv):
        """At least some long entries should occur in a trending dataset."""
        strat = MeanReversion(window=20, entry_threshold=2.0, exit_threshold=0.5)
        signals = strat.generate_signals(processed_ohlcv)
        # Should have at least one entry in 500+ days of data
        assert signals["Signal"].sum() > 0, "Expected at least one long signal"

    def test_z_score_column_present(self, processed_ohlcv):
        strat = MeanReversion(window=20)
        signals = strat.generate_signals(processed_ohlcv)
        assert "Z_Score" in signals.columns


# ── Momentum ───────────────────────────────────────────────────────────────────

class TestMomentum:
    def test_basic_signals(self, processed_ohlcv):
        strat = Momentum(lookback=60, skip_days=1)
        signals = strat.generate_signals(processed_ohlcv)
        _assert_valid_output(signals, processed_ohlcv)

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            Momentum(lookback=5)  # Below minimum 10

    def test_skip_cannot_exceed_lookback(self):
        with pytest.raises(ValueError):
            Momentum(lookback=20, skip_days=25)

    def test_trailing_return_present(self, processed_ohlcv):
        strat = Momentum(lookback=60)
        signals = strat.generate_signals(processed_ohlcv)
        assert "Trailing_Return" in signals.columns

    def test_warm_up_is_flat(self, processed_ohlcv):
        """First lookback rows must be flat (trailing return not yet available)."""
        lookback = 60
        strat = Momentum(lookback=lookback, skip_days=0)
        signals = strat.generate_signals(processed_ohlcv)
        warm_up = signals["Signal"].iloc[:lookback]
        assert (warm_up == 0).all()


# ── Buy and Hold ───────────────────────────────────────────────────────────────

class TestBuyAndHold:
    def test_always_long(self, processed_ohlcv):
        """After warm-up, position should always be 1."""
        strat = BuyAndHold()
        signals = strat.generate_signals(processed_ohlcv)
        # Signal is always 1
        assert (signals["Signal"] == 1).all()

    def test_no_parameters_needed(self):
        strat = BuyAndHold()
        assert strat.parameters == {}

    def test_returns_match_asset(self, processed_ohlcv):
        """Strategy return should match asset return (position = 1 always, shifted 1 day)."""
        strat = BuyAndHold()
        signals = strat.generate_signals(processed_ohlcv)
        # Strategy return = position (1) * asset return, but first position is 0 (shift)
        assert signals["Strategy_Return"].iloc[0] == 0.0  # First day position is 0
        # After first day, strategy_return == asset_return
        expected = signals["Position"] * signals["Asset_Return"]
        pd.testing.assert_series_equal(
            signals["Strategy_Return"], expected, check_names=False
        )
