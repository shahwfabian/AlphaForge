"""
Tests for data ingestion and preprocessing.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.data.preprocessing import preprocess, align_dataframes
from src.data.validation import (
    validate_ticker,
    validate_date_range,
    validate_dataframe,
    ValidationError,
)


# ── Preprocessing tests ────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_return_columns_added(self, sample_ohlcv):
        """Preprocessing should add Return, Log_Return, Rolling_Volatility columns."""
        df = preprocess(sample_ohlcv, ticker="TEST", save=False)
        assert "Return" in df.columns
        assert "Log_Return" in df.columns
        assert "Rolling_Volatility" in df.columns

    def test_no_nan_returns(self, sample_ohlcv):
        """After preprocessing, Return column should have no NaN (first row dropped)."""
        df = preprocess(sample_ohlcv, ticker="TEST", save=False)
        assert df["Return"].isna().sum() == 0

    def test_returns_magnitude(self, sample_ohlcv):
        """Daily returns should be reasonable — not 10x or -100%."""
        df = preprocess(sample_ohlcv, ticker="TEST", save=False)
        assert df["Return"].abs().max() < 0.50, "Suspicious return magnitude > 50%"

    def test_forward_fill_handles_gaps(self, sample_ohlcv):
        """Should forward-fill small gaps in Close."""
        df = sample_ohlcv.copy()
        df.iloc[5:8, df.columns.get_loc("Close")] = np.nan
        result = preprocess(df, ticker="TEST", save=False)
        assert result["Close"].isna().sum() == 0

    def test_duplicate_rows_removed(self, sample_ohlcv):
        """Duplicate index rows should be deduplicated."""
        df_with_dupes = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:5]])
        df_with_dupes.sort_index(inplace=True)
        result = preprocess(df_with_dupes, ticker="TEST", save=False)
        assert result.index.is_unique

    def test_rolling_vol_not_all_nan(self, sample_ohlcv):
        """Rolling volatility should have non-NaN values after warm-up."""
        df = preprocess(sample_ohlcv, ticker="TEST", save=False)
        non_nan = df["Rolling_Volatility"].dropna()
        assert len(non_nan) > 0

    def test_align_dataframes(self, sample_ohlcv):
        """align_dataframes should produce a wide multi-ticker DataFrame."""
        df1 = preprocess(sample_ohlcv, ticker="A", save=False)
        df2 = preprocess(sample_ohlcv.copy(), ticker="B", save=False)
        aligned = align_dataframes({"A": df1, "B": df2}, column="Close")
        assert "A" in aligned.columns
        assert "B" in aligned.columns
        assert not aligned.empty


# ── Validation tests ───────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_ticker(self):
        assert validate_ticker("aapl") == "AAPL"

    def test_empty_ticker_raises(self):
        with pytest.raises(ValidationError):
            validate_ticker("")

    def test_whitespace_ticker_raises(self):
        with pytest.raises(ValidationError):
            validate_ticker("   ")

    def test_valid_date_range(self):
        validate_date_range("2020-01-01", "2023-12-31")  # Should not raise

    def test_reversed_dates_raise(self):
        with pytest.raises(ValidationError):
            validate_date_range("2023-01-01", "2020-01-01")

    def test_same_dates_raise(self):
        with pytest.raises(ValidationError):
            validate_date_range("2023-01-01", "2023-01-01")

    def test_range_too_short_raises(self):
        with pytest.raises(ValidationError):
            validate_date_range("2023-01-01", "2023-01-15")  # Only 14 days

    def test_valid_dataframe(self, sample_ohlcv):
        validate_dataframe(sample_ohlcv, ticker="TEST")  # Should not raise

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValidationError):
            validate_dataframe(pd.DataFrame(), ticker="TEST")

    def test_missing_columns_raises(self, sample_ohlcv):
        df_partial = sample_ohlcv.drop(columns=["Close"])
        with pytest.raises(ValidationError):
            validate_dataframe(df_partial, ticker="TEST")
