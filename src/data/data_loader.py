"""
Market data ingestion module using yfinance.
Downloads OHLCV data, handles errors gracefully, and persists raw data.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_market_data(
    tickers: Union[str, list[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for one or more tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list of str
        Ticker symbol(s) to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    interval : str
        Data frequency ('1d', '1wk', '1mo').
    cache : bool
        If True, save raw data to disk and load from cache if available.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> raw OHLCV DataFrame. Missing/invalid tickers
        are excluded from the result with a logged warning.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ticker = ticker.upper().strip()
        cache_path = RAW_DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.parquet"

        if cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info("Loaded %s from cache.", ticker)
                result[ticker] = df
                continue
            except Exception as e:
                logger.warning("Cache read failed for %s: %s. Re-downloading.", ticker, e)

        df = _download_ticker(ticker, start_date, end_date, interval)
        if df is not None:
            if cache:
                try:
                    df.to_parquet(cache_path)
                except Exception as e:
                    logger.warning("Could not cache %s: %s", ticker, e)
            result[ticker] = df

    return result


def _download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Download a single ticker; return None on failure."""
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("No data returned for ticker: %s", ticker)
            return None

        # Flatten MultiIndex columns produced by yfinance >=0.2
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "Date"

        # Standardise column names
        raw.columns = [c.strip().replace(" ", "_") for c in raw.columns]

        # Keep only standard OHLCV columns when present
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        raw = raw[keep].copy()

        logger.info("Downloaded %d rows for %s.", len(raw), ticker)
        return raw

    except Exception as e:
        logger.error("Failed to download %s: %s", ticker, e)
        return None


def get_ticker_info(ticker: str) -> dict:
    """Return basic info dict for a ticker (name, sector, etc.)."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "industry": "N/A",
                "market_cap": None, "currency": "USD"}
