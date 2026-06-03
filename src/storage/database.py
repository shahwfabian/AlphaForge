"""
SQLite persistence layer.

Stores backtest run metadata and metrics so results can be compared
across runs without re-running the full simulation.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "alphaforge.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialise_db(db_path: Path = DB_PATH) -> None:
    """
    Create database tables if they don't already exist.

    Tables
    ------
    backtest_runs : One row per backtest run (metadata + metrics).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                benchmark    TEXT NOT NULL DEFAULT 'SPY',
                strategy     TEXT NOT NULL,
                parameters   TEXT,          -- JSON
                start_date   TEXT NOT NULL,
                end_date     TEXT NOT NULL,
                initial_capital  REAL NOT NULL,
                transaction_cost REAL NOT NULL,
                slippage         REAL NOT NULL,
                cumulative_return   REAL,
                annualised_return   REAL,
                sharpe_ratio        REAL,
                sortino_ratio       REAL,
                calmar_ratio        REAL,
                max_drawdown        REAL,
                annualised_vol      REAL,
                win_rate            REAL,
                profit_factor       REAL,
                alpha               REAL,
                beta                REAL,
                var_95              REAL,
                expected_shortfall  REAL,
                total_days          INTEGER
            )
        """)
        conn.commit()
        logger.info("Database initialised at %s", db_path)
    finally:
        conn.close()


def save_run(
    ticker: str,
    benchmark: str,
    strategy_name: str,
    parameters: dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
    metrics: dict,
    db_path: Path = DB_PATH,
) -> int:
    """
    Insert a backtest run record and return the new run_id.

    Parameters
    ----------
    metrics : dict
        Combined performance + risk + benchmark metrics dict.

    Returns
    -------
    int
        Auto-incremented run_id.
    """
    initialise_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO backtest_runs (
                timestamp, ticker, benchmark, strategy, parameters,
                start_date, end_date, initial_capital, transaction_cost, slippage,
                cumulative_return, annualised_return, sharpe_ratio, sortino_ratio,
                calmar_ratio, max_drawdown, annualised_vol, win_rate, profit_factor,
                alpha, beta, var_95, expected_shortfall, total_days
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                ticker,
                benchmark,
                strategy_name,
                json.dumps(parameters),
                start_date,
                end_date,
                initial_capital,
                transaction_cost,
                slippage,
                metrics.get("Cumulative_Return"),
                metrics.get("Annualised_Return"),
                metrics.get("Sharpe_Ratio"),
                metrics.get("Sortino_Ratio"),
                metrics.get("Calmar_Ratio"),
                metrics.get("Max_Drawdown"),
                metrics.get("Annualised_Volatility"),
                metrics.get("Win_Rate"),
                metrics.get("Profit_Factor"),
                metrics.get("Alpha"),
                metrics.get("Beta"),
                metrics.get("VaR_Historical_95"),
                metrics.get("Expected_Shortfall_95"),
                metrics.get("Total_Days"),
            ),
        )
        conn.commit()
        run_id = cursor.lastrowid
        logger.info("Saved run %d to database.", run_id)
        return run_id
    finally:
        conn.close()


def load_runs(db_path: Path = DB_PATH, limit: int = 50) -> pd.DataFrame:
    """
    Retrieve recent backtest runs from the database.

    Parameters
    ----------
    limit : int
        Maximum number of rows to return (most recent first).

    Returns
    -------
    pd.DataFrame
    """
    initialise_db(db_path)
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM backtest_runs ORDER BY run_id DESC LIMIT {limit}",
            conn,
        )
        return df
    finally:
        conn.close()


def delete_run(run_id: int, db_path: Path = DB_PATH) -> None:
    """Delete a single backtest run by run_id."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM backtest_runs WHERE run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()
