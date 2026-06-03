"""
AlphaForge — Asset Universe
============================
Provides search, validation, and metadata lookup over a curated local
CSV of ~400 tradeable U.S. assets (stocks, ETFs, bonds, commodities).

The CSV lives at  data/asset_universe.csv  relative to the project root.
You can extend the universe at any time by appending rows to that file —
no code changes required.

Public API
----------
load_asset_universe()               → pd.DataFrame
search_assets(query, types, limit)  → list[dict]
validate_symbol(symbol)             → bool
get_asset_metadata(symbol)          → dict | None
parse_symbol_from_label(label)      → str
get_all_labels(asset_types)         → list[str]
get_asset_types()                   → list[str]
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
_UNIVERSE_PATH = _PROJECT_ROOT / "data" / "asset_universe.csv"

# Required CSV columns
_REQUIRED_COLS = {"symbol", "name", "asset_type", "exchange", "sector", "currency"}

# Canonical asset type display order for UI filter
ASSET_TYPE_ORDER = [
    "Stock",
    "Index ETF",
    "Sector ETF",
    "Bond ETF",
    "Commodity ETF",
    "Thematic ETF",
    "Leveraged ETF",
    "Crypto ETF",
]


# ── Core loader ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_asset_universe() -> pd.DataFrame:
    """
    Load and cache the asset universe CSV.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, name, asset_type, exchange, sector, currency, label.
        Indexed 0…N-1.  Always returns a non-None DataFrame (empty on error).

    Notes
    -----
    Results are cached in-process via ``lru_cache``.  To force a reload
    (e.g. after editing the CSV) call ``load_asset_universe.cache_clear()``.
    """
    if not _UNIVERSE_PATH.exists():
        logger.error("Asset universe CSV not found at %s", _UNIVERSE_PATH)
        return _empty_universe()

    try:
        df = pd.read_csv(_UNIVERSE_PATH, dtype=str, comment="#")
    except Exception as exc:
        logger.error("Failed to read asset universe: %s", exc)
        return _empty_universe()

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error("Asset universe CSV is missing columns: %s", missing)
        return _empty_universe()

    # Clean string values
    for col in _REQUIRED_COLS:
        df[col] = df[col].fillna("").str.strip()

    df["symbol"] = df["symbol"].str.upper()

    # Drop rows with blank symbol or name
    df = df[df["symbol"].ne("") & df["name"].ne("")].reset_index(drop=True)

    # Pre-compute formatted label for each row
    df["label"] = df.apply(
        lambda r: f"{r['symbol']} — {r['name']} ({r['asset_type']})", axis=1
    )

    logger.debug("Loaded asset universe: %d assets", len(df))
    return df


def _empty_universe() -> pd.DataFrame:
    cols = list(_REQUIRED_COLS) + ["label"]
    return pd.DataFrame(columns=cols)


# ── Search ────────────────────────────────────────────────────────────────────

def search_assets(
    query: str,
    asset_types: list[str] | None = None,
    limit: int = 25,
) -> list[dict]:
    """
    Search the universe by ticker or company/fund name.

    Priority ranking:
    1. Exact ticker match          (e.g. "SPY" → SPY)
    2. Ticker starts with query    (e.g. "SP"  → SPY, SPGI …)
    3. Name starts with query      (e.g. "app" → Apple Inc.)
    4. Ticker contains query       (e.g. "LD"  → GLD, GLDM …)
    5. Name contains query         (e.g. "gold" → all Gold funds)

    Parameters
    ----------
    query : str
        Search string — ticker or company name fragment.
    asset_types : list[str] | None
        Optional whitelist of asset types (e.g. ["Stock", "Bond ETF"]).
        Passing ``None`` or ``[]`` returns all types.
    limit : int
        Maximum results to return.

    Returns
    -------
    list[dict]
        Each dict has: symbol, name, asset_type, exchange, sector,
        currency, label.
    """
    df = load_asset_universe()

    if asset_types:
        df = df[df["asset_type"].isin(asset_types)].copy()

    if df.empty:
        return []

    q = query.strip() if query else ""

    if not q:
        return df.head(limit).to_dict("records")

    q_up    = q.upper()
    q_lo    = q.lower()

    sym     = df["symbol"]
    nam_lo  = df["name"].str.lower()

    mask_exact_sym   = sym == q_up
    mask_prefix_sym  = sym.str.startswith(q_up) & ~mask_exact_sym
    mask_name_starts = nam_lo.str.startswith(q_lo) & ~mask_exact_sym & ~mask_prefix_sym
    mask_sym_contains = (
        sym.str.contains(q_up, na=False, regex=False)
        & ~mask_exact_sym & ~mask_prefix_sym
    )
    mask_name_contains = (
        nam_lo.str.contains(q_lo, na=False, regex=False)
        & ~mask_exact_sym & ~mask_prefix_sym
        & ~mask_name_starts & ~mask_sym_contains
    )

    ordered = pd.concat([
        df[mask_exact_sym],
        df[mask_prefix_sym],
        df[mask_name_starts],
        df[mask_sym_contains],
        df[mask_name_contains],
    ]).drop_duplicates(subset="symbol")

    return ordered.head(limit).to_dict("records")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_symbol(symbol: str) -> bool:
    """
    Return True if ``symbol`` exists in the local asset universe.

    Note: a symbol not in the universe may still be valid on yfinance —
    use this only as a soft check, not a hard gate.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    df = load_asset_universe()
    return symbol.upper().strip() in df["symbol"].values


# ── Metadata ─────────────────────────────────────────────────────────────────

def get_asset_metadata(symbol: str) -> dict | None:
    """
    Return the full metadata row for ``symbol`` as a dict, or ``None``
    if the symbol is not in the universe.

    Keys: symbol, name, asset_type, exchange, sector, currency, label.
    """
    if not symbol or not isinstance(symbol, str):
        return None
    df = load_asset_universe()
    rows = df[df["symbol"] == symbol.upper().strip()]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ── Label helpers ─────────────────────────────────────────────────────────────

def parse_symbol_from_label(label: str) -> str:
    """
    Extract the ticker from a formatted label.

    >>> parse_symbol_from_label("AAPL — Apple Inc. (Stock)")
    'AAPL'
    >>> parse_symbol_from_label("SPY — SPDR S&P 500 ETF Trust (Index ETF)")
    'SPY'
    """
    if not label or " — " not in label:
        return label.strip().upper() if label else ""
    return label.split(" — ")[0].strip().upper()


def get_all_labels(asset_types: list[str] | None = None) -> list[str]:
    """
    Return all formatted labels, optionally filtered by asset type(s).

    Used to populate Streamlit multiselect / selectbox options.
    """
    df = load_asset_universe()
    if asset_types:
        df = df[df["asset_type"].isin(asset_types)]
    return df["label"].tolist()


def get_asset_types() -> list[str]:
    """Return asset types present in the universe, in canonical display order."""
    df = load_asset_universe()
    present = set(df["asset_type"].unique())
    ordered = [t for t in ASSET_TYPE_ORDER if t in present]
    # Append any types not in the canonical order
    ordered += sorted(present - set(ordered))
    return ordered


def get_label_for_symbol(symbol: str) -> str | None:
    """Return the formatted label for ``symbol``, or ``None`` if not found."""
    meta = get_asset_metadata(symbol)
    return meta["label"] if meta else None
