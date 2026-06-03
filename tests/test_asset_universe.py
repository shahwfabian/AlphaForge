"""
Tests for src/data/asset_universe.py

Covers:
- load_asset_universe
- search_assets (exact, prefix, name, case-insensitive, type filter)
- validate_symbol
- get_asset_metadata
- parse_symbol_from_label
- get_all_labels
- get_asset_types
"""

import pytest
import pandas as pd

# Clear the lru_cache before each test to ensure isolation
from src.data import asset_universe as _au_mod


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the load_asset_universe LRU cache before every test."""
    _au_mod.load_asset_universe.cache_clear()
    yield
    _au_mod.load_asset_universe.cache_clear()


from src.data.asset_universe import (
    load_asset_universe,
    search_assets,
    validate_symbol,
    get_asset_metadata,
    parse_symbol_from_label,
    get_all_labels,
    get_asset_types,
    get_label_for_symbol,
    ASSET_TYPE_ORDER,
)


# ── load_asset_universe ────────────────────────────────────────────────────────

class TestLoadAssetUniverse:
    def test_returns_dataframe(self):
        df = load_asset_universe()
        assert isinstance(df, pd.DataFrame)

    def test_not_empty(self):
        df = load_asset_universe()
        assert len(df) > 100, "Universe should have at least 100 assets"

    def test_has_required_columns(self):
        df = load_asset_universe()
        for col in ("symbol", "name", "asset_type", "exchange", "sector", "currency", "label"):
            assert col in df.columns, f"Missing column: {col}"

    def test_symbols_are_uppercase(self):
        df = load_asset_universe()
        assert df["symbol"].str.isupper().all(), "All symbols should be uppercase"

    def test_no_blank_symbols(self):
        df = load_asset_universe()
        assert (df["symbol"].str.strip() != "").all()

    def test_label_column_format(self):
        df = load_asset_universe()
        # Labels should contain " — " separator and end with "(Type)"
        sample = df["label"].iloc[0]
        assert " — " in sample
        assert sample.endswith(")")

    def test_known_tickers_present(self):
        df = load_asset_universe()
        symbols = set(df["symbol"].values)
        for expected in ("AAPL", "MSFT", "SPY", "QQQ", "TLT", "GLD"):
            assert expected in symbols, f"{expected} missing from universe"

    def test_caching_returns_same_object(self):
        df1 = load_asset_universe()
        df2 = load_asset_universe()
        assert df1 is df2, "Should return cached object on second call"


# ── search_assets ─────────────────────────────────────────────────────────────

class TestSearchAssets:
    def test_exact_ticker_match_is_first(self):
        results = search_assets("SPY")
        assert results, "Should return at least one result"
        assert results[0]["symbol"] == "SPY"

    def test_exact_ticker_match_aapl(self):
        results = search_assets("AAPL")
        assert results[0]["symbol"] == "AAPL"

    def test_partial_ticker_prefix(self):
        results = search_assets("QQ")
        symbols = [r["symbol"] for r in results]
        assert "QQQ" in symbols

    def test_partial_ticker_prefix_tl(self):
        results = search_assets("TL")
        symbols = [r["symbol"] for r in results]
        assert "TLT" in symbols

    def test_company_name_search(self):
        results = search_assets("apple")
        symbols = [r["symbol"] for r in results]
        assert "AAPL" in symbols

    def test_company_name_partial(self):
        results = search_assets("microsoft")
        symbols = [r["symbol"] for r in results]
        assert "MSFT" in symbols

    def test_case_insensitive_upper(self):
        results_lower = search_assets("apple")
        results_upper = search_assets("APPLE")
        syms_lower = {r["symbol"] for r in results_lower}
        syms_upper = {r["symbol"] for r in results_upper}
        assert "AAPL" in syms_lower
        assert "AAPL" in syms_upper

    def test_case_insensitive_mixed(self):
        results = search_assets("ApPlE")
        assert any(r["symbol"] == "AAPL" for r in results)

    def test_fund_name_search(self):
        results = search_assets("gold")
        symbols = [r["symbol"] for r in results]
        # GLD or IAU or GDX should appear
        assert any(s in symbols for s in ("GLD", "IAU", "GDX"))

    def test_asset_type_filter_stock_only(self):
        results = search_assets("", asset_types=["Stock"])
        assert results
        assert all(r["asset_type"] == "Stock" for r in results)

    def test_asset_type_filter_bond_etf(self):
        results = search_assets("", asset_types=["Bond ETF"])
        assert results
        assert all(r["asset_type"] == "Bond ETF" for r in results)

    def test_asset_type_filter_multiple(self):
        results = search_assets("", asset_types=["Index ETF", "Sector ETF"])
        for r in results:
            assert r["asset_type"] in ("Index ETF", "Sector ETF")

    def test_asset_type_filter_with_query(self):
        results = search_assets("tech", asset_types=["Sector ETF"])
        assert results
        assert all(r["asset_type"] == "Sector ETF" for r in results)

    def test_limit_parameter(self):
        results = search_assets("", limit=5)
        assert len(results) <= 5

    def test_limit_default(self):
        results = search_assets("a")
        assert len(results) <= 25

    def test_empty_query_returns_results(self):
        results = search_assets("")
        assert len(results) > 0

    def test_no_match_returns_empty(self):
        results = search_assets("XYZXYZXYZ_NOMATCH_12345")
        assert results == []

    def test_result_dict_has_required_keys(self):
        results = search_assets("SPY", limit=1)
        assert results
        for key in ("symbol", "name", "asset_type", "exchange", "sector", "currency", "label"):
            assert key in results[0], f"Key '{key}' missing from result"

    def test_exact_match_ranked_above_prefix(self):
        # "SPY" should rank above "SPYI" or "SPYG" if they exist
        results = search_assets("SPY")
        assert results[0]["symbol"] == "SPY"

    def test_spy_found_by_name_fragment(self):
        results = search_assets("SPDR S&P 500")
        symbols = [r["symbol"] for r in results]
        assert "SPY" in symbols


# ── validate_symbol ───────────────────────────────────────────────────────────

class TestValidateSymbol:
    def test_known_symbol_returns_true(self):
        assert validate_symbol("AAPL") is True

    def test_known_etf_returns_true(self):
        assert validate_symbol("SPY") is True

    def test_unknown_symbol_returns_false(self):
        assert validate_symbol("XYZNOTREAL999") is False

    def test_case_insensitive_lower(self):
        assert validate_symbol("aapl") is True

    def test_case_insensitive_mixed(self):
        assert validate_symbol("AaPl") is True

    def test_empty_string_returns_false(self):
        assert validate_symbol("") is False

    def test_none_returns_false(self):
        assert validate_symbol(None) is False  # type: ignore[arg-type]

    def test_whitespace_stripped(self):
        assert validate_symbol("  AAPL  ") is True


# ── get_asset_metadata ────────────────────────────────────────────────────────

class TestGetAssetMetadata:
    def test_known_symbol_returns_dict(self):
        meta = get_asset_metadata("AAPL")
        assert isinstance(meta, dict)

    def test_unknown_symbol_returns_none(self):
        meta = get_asset_metadata("XYZNOTREAL999")
        assert meta is None

    def test_metadata_has_required_keys(self):
        meta = get_asset_metadata("SPY")
        for key in ("symbol", "name", "asset_type", "exchange", "sector", "currency"):
            assert key in meta, f"Key '{key}' missing from metadata"

    def test_aapl_is_stock(self):
        meta = get_asset_metadata("AAPL")
        assert meta["asset_type"] == "Stock"

    def test_spy_is_index_etf(self):
        meta = get_asset_metadata("SPY")
        assert meta["asset_type"] == "Index ETF"

    def test_tlt_is_bond_etf(self):
        meta = get_asset_metadata("TLT")
        assert meta["asset_type"] == "Bond ETF"

    def test_gld_is_commodity_etf(self):
        meta = get_asset_metadata("GLD")
        assert meta["asset_type"] == "Commodity ETF"

    def test_case_insensitive(self):
        meta_upper = get_asset_metadata("MSFT")
        meta_lower = get_asset_metadata("msft")
        assert meta_upper is not None
        assert meta_lower is not None
        assert meta_upper["symbol"] == meta_lower["symbol"]

    def test_label_key_present(self):
        meta = get_asset_metadata("NVDA")
        assert "label" in meta
        assert "NVDA" in meta["label"]


# ── parse_symbol_from_label ───────────────────────────────────────────────────

class TestParseSymbolFromLabel:
    def test_stock_label(self):
        assert parse_symbol_from_label("AAPL — Apple Inc. (Stock)") == "AAPL"

    def test_etf_label(self):
        assert parse_symbol_from_label("SPY — SPDR S&P 500 ETF Trust (Index ETF)") == "SPY"

    def test_bond_etf_label(self):
        assert parse_symbol_from_label("TLT — iShares 20+ Year Treasury Bond ETF (Bond ETF)") == "TLT"

    def test_commodity_etf_label(self):
        assert parse_symbol_from_label("GLD — SPDR Gold Shares (Commodity ETF)") == "GLD"

    def test_strips_whitespace(self):
        assert parse_symbol_from_label("  MSFT — Microsoft Corporation (Stock)  ") == "MSFT"

    def test_plain_ticker_passthrough(self):
        # If label has no separator, return as-is (upper)
        assert parse_symbol_from_label("AAPL") == "AAPL"

    def test_lowercase_ticker_normalised(self):
        assert parse_symbol_from_label("aapl — Apple Inc. (Stock)") == "AAPL"

    def test_multi_word_company_name(self):
        assert parse_symbol_from_label(
            "JPM — JPMorgan Chase & Co. (Stock)"
        ) == "JPM"


# ── get_all_labels ─────────────────────────────────────────────────────────────

class TestGetAllLabels:
    def test_returns_list(self):
        labels = get_all_labels()
        assert isinstance(labels, list)

    def test_not_empty(self):
        labels = get_all_labels()
        assert len(labels) > 100

    def test_labels_are_strings(self):
        labels = get_all_labels()
        assert all(isinstance(l, str) for l in labels)

    def test_filter_by_bond_etf(self):
        labels = get_all_labels(["Bond ETF"])
        assert labels
        assert all("Bond ETF" in l for l in labels)

    def test_filter_by_stock(self):
        labels = get_all_labels(["Stock"])
        assert labels
        assert all("Stock" in l for l in labels)

    def test_aapl_label_in_stock_filter(self):
        labels = get_all_labels(["Stock"])
        assert any("AAPL" in l for l in labels)

    def test_spy_label_in_index_filter(self):
        labels = get_all_labels(["Index ETF"])
        assert any("SPY" in l for l in labels)


# ── get_asset_types ───────────────────────────────────────────────────────────

class TestGetAssetTypes:
    def test_returns_list(self):
        types = get_asset_types()
        assert isinstance(types, list)

    def test_expected_types_present(self):
        types = get_asset_types()
        for expected in ("Stock", "Index ETF", "Bond ETF", "Commodity ETF"):
            assert expected in types, f"Expected asset type '{expected}' not found"

    def test_canonical_order_respected(self):
        types   = get_asset_types()
        present = [t for t in ASSET_TYPE_ORDER if t in types]
        indices = [types.index(t) for t in present]
        assert indices == sorted(indices), "Canonical order not respected"


# ── get_label_for_symbol ──────────────────────────────────────────────────────

class TestGetLabelForSymbol:
    def test_known_symbol(self):
        label = get_label_for_symbol("AAPL")
        assert label is not None
        assert "AAPL" in label
        assert "Apple" in label

    def test_unknown_symbol(self):
        assert get_label_for_symbol("XYZNOTREAL") is None

    def test_label_includes_asset_type(self):
        label = get_label_for_symbol("SPY")
        assert "Index ETF" in label
