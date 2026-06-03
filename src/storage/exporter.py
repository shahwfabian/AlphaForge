"""
CSV export utilities.

Writes strategy results (metrics, trade log, portfolio history)
to the /exports directory for download via the Streamlit dashboard.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPORTS_DIR = Path(__file__).resolve().parents[2] / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_prefix(ticker: str, strategy: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_strategy = strategy.replace(" ", "_").lower()
    return f"{ticker}_{safe_strategy}_{ts}"


def export_metrics(metrics: dict, ticker: str, strategy: str) -> Path:
    """
    Export performance + risk + benchmark metrics to CSV.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    prefix = _run_prefix(ticker, strategy)
    out_path = EXPORTS_DIR / f"{prefix}_metrics.csv"

    rows = [{"Metric": k, "Value": v} for k, v in metrics.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    logger.info("Metrics exported to %s", out_path)
    return out_path


def export_trade_log(trade_df: pd.DataFrame, ticker: str, strategy: str) -> Path:
    """
    Export the full trade log to CSV.

    Returns
    -------
    Path
    """
    prefix = _run_prefix(ticker, strategy)
    out_path = EXPORTS_DIR / f"{prefix}_trades.csv"
    trade_df.to_csv(out_path, index=False)
    logger.info("Trade log exported to %s", out_path)
    return out_path


def export_portfolio(equity_df: pd.DataFrame, ticker: str, strategy: str) -> Path:
    """
    Export the daily portfolio equity curve to CSV.

    Returns
    -------
    Path
    """
    prefix = _run_prefix(ticker, strategy)
    out_path = EXPORTS_DIR / f"{prefix}_portfolio.csv"
    equity_df.to_csv(out_path)
    logger.info("Portfolio history exported to %s", out_path)
    return out_path


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for Streamlit's st.download_button.

    Usage
    -----
    st.download_button("Download", dataframe_to_csv_bytes(df), "file.csv", "text/csv")
    """
    return df.to_csv(index=True).encode("utf-8")


def metrics_to_csv_bytes(metrics: dict) -> bytes:
    """Convert a metrics dict to CSV bytes."""
    rows = [{"Metric": k, "Value": v} for k, v in metrics.items()]
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")
