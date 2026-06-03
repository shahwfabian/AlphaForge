"""
Shared pytest fixtures for AlphaForge tests.
"""

import numpy as np
import pandas as pd
import pytest

TRADING_DAYS = 504   # ~2 years of data


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame with a realistic-looking price series.
    Uses a Geometric Brownian Motion path so strategies produce real signals.
    """
    rng = np.random.default_rng(42)
    n = TRADING_DAYS
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    # GBM: daily return ~ N(0.0003, 0.012)
    daily_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = 100.0 * np.cumprod(1 + daily_returns)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def processed_ohlcv(sample_ohlcv) -> pd.DataFrame:
    """Preprocessed version of sample_ohlcv with Return and Log_Return columns."""
    from src.data.preprocessing import preprocess
    return preprocess(sample_ohlcv, ticker="TEST", save=False)


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """Flat price series (no movement) — useful for edge-case tests."""
    n = 100
    dates = pd.date_range(start="2021-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [100.5] * n,
            "Low": [99.5] * n,
            "Close": [100.0] * n,
            "Volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
