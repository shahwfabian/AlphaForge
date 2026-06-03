"""
Input validation utilities for tickers, dates, and DataFrames.
"""

import logging
from datetime import datetime
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLS = {"Open", "High", "Low", "Close", "Volume"}
MIN_ROWS = 30  # Minimum number of rows to be usable for strategy signals


class ValidationError(ValueError):
    """Raised when a validation check fails."""


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalise a single ticker symbol.

    Parameters
    ----------
    ticker : str
        Raw ticker input from user.

    Returns
    -------
    str
        Uppercased, stripped ticker.

    Raises
    ------
    ValidationError
        If the ticker is empty or contains invalid characters.
    """
    if not isinstance(ticker, str):
        raise ValidationError(f"Ticker must be a string, got {type(ticker).__name__}.")
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValidationError("Ticker symbol cannot be empty.")
    if len(ticker) > 10:
        raise ValidationError(f"Ticker '{ticker}' seems too long (max 10 chars).")
    return ticker


def validate_tickers(tickers: Union[str, list]) -> list[str]:
    """Validate and normalise a list of tickers."""
    if isinstance(tickers, str):
        tickers = [tickers]
    return [validate_ticker(t) for t in tickers]


def validate_date(date_str: str, label: str = "date") -> str:
    """
    Validate a date string in 'YYYY-MM-DD' format.

    Parameters
    ----------
    date_str : str
        Date string to validate.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    str
        The validated date string.

    Raises
    ------
    ValidationError
        If the format is invalid.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValidationError(
            f"Invalid {label} '{date_str}'. Expected format: YYYY-MM-DD."
        )
    return date_str


def validate_date_range(start_date: str, end_date: str) -> None:
    """
    Ensure start_date < end_date and the range is at least 60 days.

    Raises
    ------
    ValidationError
    """
    validate_date(start_date, "start_date")
    validate_date(end_date, "end_date")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if start >= end:
        raise ValidationError("start_date must be strictly before end_date.")

    delta_days = (end - start).days
    if delta_days < 60:
        raise ValidationError(
            f"Date range is only {delta_days} days. "
            "Please use at least 60 days for meaningful backtesting."
        )


def validate_dataframe(df: pd.DataFrame, ticker: str = "unknown") -> None:
    """
    Check that a DataFrame has the expected OHLCV columns and sufficient rows.

    Raises
    ------
    ValidationError
    """
    if df is None or df.empty:
        raise ValidationError(f"DataFrame for '{ticker}' is empty.")

    missing = REQUIRED_OHLCV_COLS - set(df.columns)
    if missing:
        raise ValidationError(
            f"DataFrame for '{ticker}' is missing columns: {missing}"
        )

    if len(df) < MIN_ROWS:
        raise ValidationError(
            f"DataFrame for '{ticker}' has only {len(df)} rows. "
            f"Minimum required: {MIN_ROWS}."
        )


def validate_capital(capital: float) -> None:
    """Ensure initial capital is a positive number."""
    if not isinstance(capital, (int, float)) or capital <= 0:
        raise ValidationError(f"Initial capital must be a positive number, got {capital}.")


def validate_cost_params(transaction_cost: float, slippage: float) -> None:
    """Validate transaction cost and slippage are in [0, 0.05]."""
    for name, val in [("transaction_cost", transaction_cost), ("slippage", slippage)]:
        if not (0.0 <= val <= 0.05):
            raise ValidationError(
                f"{name} must be between 0 and 0.05 (0–5%). Got {val}."
            )


def check_sufficient_data(df: pd.DataFrame, min_periods: int, ticker: str = "") -> bool:
    """
    Return True if df has at least min_periods rows; log warning otherwise.
    Does not raise — callers decide whether to abort.
    """
    if len(df) < min_periods:
        logger.warning(
            "%s: only %d rows available but strategy needs %d. "
            "Consider extending the date range.",
            ticker, len(df), min_periods
        )
        return False
    return True
