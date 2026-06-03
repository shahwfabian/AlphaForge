"""
AlphaForge — Strategy Configuration Save / Load
================================================
Persists user-defined backtest configurations (assets, strategy,
parameters, date range, capital, etc.) to a SQLite database so they
can be recalled across sessions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "strategy_configs.db"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_configs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            config_json TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        )
    """)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def save_config(
    name: str,
    config: dict,
    db_path: Path | None = None,
) -> int:
    """
    Save a strategy configuration.

    The ``config`` dict should contain serialisable values such as:
    tickers, benchmark, strategy, strategy_params, start_date, end_date,
    initial_capital, transaction_cost, slippage, allocation_weights.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "SPY MA20/50 2020-2024").
    config : dict
        Configuration dict from ``render_sidebar()`` or user code.
    db_path : Path | None
        Override DB location (used in tests).

    Returns
    -------
    int
        Auto-generated row ID of the new record.
    """
    # Strip non-serialisable keys
    safe = {k: v for k, v in config.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}

    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO strategy_configs (name, config_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (name.strip(), json.dumps(safe, default=str), now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def load_config(
    config_id: int,
    db_path: Path | None = None,
) -> dict | None:
    """
    Load a configuration by integer ID.

    Returns
    -------
    dict | None
        Deserialised config, or ``None`` if not found.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT config_json FROM strategy_configs WHERE id = ?",
            (config_id,),
        ).fetchone()

    if row is None:
        return None
    try:
        return json.loads(row["config_json"])
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode config %d: %s", config_id, exc)
        return None


def list_configs(db_path: Path | None = None) -> pd.DataFrame:
    """
    Return a summary DataFrame of all saved configurations.

    Columns: id, name, created_at, updated_at.
    Sorted newest-first by updated_at.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, updated_at "
            "FROM strategy_configs ORDER BY updated_at DESC"
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["id", "name", "created_at", "updated_at"])
    return pd.DataFrame([dict(r) for r in rows])


def delete_config(config_id: int, db_path: Path | None = None) -> None:
    """Delete a saved configuration by ID."""
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM strategy_configs WHERE id = ?",
            (config_id,),
        )
        conn.commit()


def rename_config(
    config_id: int,
    new_name: str,
    db_path: Path | None = None,
) -> None:
    """Rename a saved configuration."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE strategy_configs SET name = ?, updated_at = ? WHERE id = ?",
            (new_name.strip(), _now(), config_id),
        )
        conn.commit()
