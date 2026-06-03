"""
Data preprocessing module.
Cleans raw OHLCV data, computes returns, log returns, and rolling volatility.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(
    df: pd.DataFrame,
    ticker: str,
    rolling_vol_window: int = 21,
    save: bool = True,
) -> pd.DataFrame:
    """
    Clean and enrich a raw OHLCV DataFrame.

    Steps:
        1. Drop fully-duplicate rows
        2. Forward-fill then back-fill small gaps in price columns
        3. Drop rows where Close is still NaN
        4. Compute simple daily return
        5. Compute log return
        6. Compute rolling annualised volatility

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame indexed by date.
    ticker : str
        Ticker symbol (used for cache file name).
    rolling_vol_window : int
        Look-back window (trading days) for rolling volatility.
    save : bool
        If True, persist the processed DataFrame to data/processed/.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with additional columns:
        ``Return``, ``Log_Return``, ``Rolling_Volatility``.
    """
    df = df.copy()
    df.sort_index(inplace=True)

    # 1. Drop exact duplicate rows
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < before:
        logger.warning("%s: removed %d duplicate rows.", ticker, before - len(df))

    # 2. Fill small gaps (up to 5 consecutive missing values)
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill(limit=5).bfill(limit=5)

    # Fill volume gaps with 0 (exchange holiday / halted trading)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    # 3. Drop rows where Close is still missing
    before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    if len(df) < before:
        logger.warning("%s: dropped %d rows with missing Close.", ticker, before - len(df))

    # 4. Simple daily return:  r_t = (P_t / P_{t-1}) - 1
    df["Return"] = df["Close"].pct_change()

    # 5. Log return:  lr_t = ln(P_t / P_{t-1})
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 6. Rolling annualised volatility: σ_annual = σ_daily * sqrt(252)
    df["Rolling_Volatility"] = (
        df["Return"].rolling(window=rolling_vol_window).std() * np.sqrt(252)
    )

    # Drop the very first row which will always have NaN returns
    df.dropna(subset=["Return"], inplace=True)

    if save:
        out_path = PROCESSED_DATA_DIR / f"{ticker}_processed.parquet"
        try:
            df.to_parquet(out_path)
            logger.info("Saved processed data for %s to %s", ticker, out_path)
        except Exception as e:
            logger.warning("Could not save processed data for %s: %s", ticker, e)

    return df


def preprocess_multiple(
    data: dict[str, pd.DataFrame],
    rolling_vol_window: int = 21,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Preprocess a dictionary of raw OHLCV DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of ticker -> raw OHLCV DataFrame.
    rolling_vol_window : int
        Look-back window for rolling volatility.
    save : bool
        Persist processed files to disk.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> processed DataFrame.
    """
    return {
        ticker: preprocess(df, ticker, rolling_vol_window, save)
        for ticker, df in data.items()
    }


def align_dataframes(
    data: dict[str, pd.DataFrame],
    column: str = "Close",
) -> pd.DataFrame:
    """
    Align multiple tickers into a single wide DataFrame on a common date index.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of ticker -> processed DataFrame.
    column : str
        Column to extract from each ticker's DataFrame.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with one column per ticker, inner-joined on dates.
    """
    frames = {ticker: df[column].rename(ticker) for ticker, df in data.items() if column in df.columns}
    if not frames:
        return pd.DataFrame()
    aligned = pd.concat(frames.values(), axis=1, join="inner")
    aligned.sort_index(inplace=True)
    return aligned


def compute_returns_matrix(aligned_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from an aligned price matrix."""
    return aligned_prices.pct_change().dropna()
