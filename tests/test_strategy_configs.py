"""Tests for src/storage/strategy_configs.py"""

import pytest
import pandas as pd
from pathlib import Path

from src.storage.strategy_configs import (
    save_config,
    load_config,
    list_configs,
    delete_config,
    rename_config,
)

SAMPLE_CONFIG = {
    "ticker":           "AAPL",
    "tickers":          ["AAPL"],
    "benchmark":        "SPY",
    "start_date":       "2020-01-01",
    "end_date":         "2023-12-31",
    "initial_capital":  100_000.0,
    "transaction_cost": 0.001,
    "slippage":         0.0005,
    "strategy":         "Moving Average Crossover",
    "strategy_params":  {"short_window": 20, "long_window": 50},
    "risk_free_rate":   0.0,
}


@pytest.fixture
def db(tmp_path):
    """Temporary SQLite database for each test."""
    return tmp_path / "test_configs.db"


class TestSaveAndLoadConfig:
    def test_save_returns_integer_id(self, db):
        rid = save_config("test", SAMPLE_CONFIG, db_path=db)
        assert isinstance(rid, int)
        assert rid > 0

    def test_load_returns_dict(self, db):
        rid = save_config("test", SAMPLE_CONFIG, db_path=db)
        cfg = load_config(rid, db_path=db)
        assert isinstance(cfg, dict)

    def test_round_trip_preserves_values(self, db):
        rid = save_config("test", SAMPLE_CONFIG, db_path=db)
        cfg = load_config(rid, db_path=db)
        assert cfg["ticker"]           == "AAPL"
        assert cfg["benchmark"]        == "SPY"
        assert cfg["initial_capital"]  == pytest.approx(100_000.0)
        assert cfg["strategy"]         == "Moving Average Crossover"

    def test_load_nonexistent_returns_none(self, db):
        result = load_config(99999, db_path=db)
        assert result is None

    def test_multiple_configs_saved_independently(self, db):
        id1 = save_config("config_1", {"ticker": "AAPL"}, db_path=db)
        id2 = save_config("config_2", {"ticker": "MSFT"}, db_path=db)
        assert id1 != id2
        assert load_config(id1, db_path=db)["ticker"] == "AAPL"
        assert load_config(id2, db_path=db)["ticker"] == "MSFT"

    def test_nested_dict_preserved(self, db):
        config = {"strategy_params": {"short_window": 10, "long_window": 30}}
        rid = save_config("nested", config, db_path=db)
        loaded = load_config(rid, db_path=db)
        assert loaded["strategy_params"]["short_window"] == 10

    def test_list_value_preserved(self, db):
        config = {"tickers": ["AAPL", "MSFT", "SPY"]}
        rid = save_config("list_test", config, db_path=db)
        loaded = load_config(rid, db_path=db)
        assert loaded["tickers"] == ["AAPL", "MSFT", "SPY"]


class TestListConfigs:
    def test_returns_dataframe(self, db):
        save_config("test", SAMPLE_CONFIG, db_path=db)
        df = list_configs(db_path=db)
        assert isinstance(df, pd.DataFrame)

    def test_empty_db_returns_empty_df(self, db):
        df = list_configs(db_path=db)
        assert df.empty

    def test_has_required_columns(self, db):
        save_config("test", SAMPLE_CONFIG, db_path=db)
        df = list_configs(db_path=db)
        for col in ("id", "name", "created_at", "updated_at"):
            assert col in df.columns

    def test_count_increases(self, db):
        save_config("c1", {}, db_path=db)
        save_config("c2", {}, db_path=db)
        df = list_configs(db_path=db)
        assert len(df) == 2

    def test_sorted_newest_first(self, db):
        id1 = save_config("first",  {}, db_path=db)
        id2 = save_config("second", {}, db_path=db)
        df  = list_configs(db_path=db)
        # Newest (largest id) should be first
        assert df.iloc[0]["id"] == id2


class TestDeleteConfig:
    def test_deleted_config_returns_none(self, db):
        rid = save_config("to_delete", SAMPLE_CONFIG, db_path=db)
        delete_config(rid, db_path=db)
        assert load_config(rid, db_path=db) is None

    def test_list_count_decreases(self, db):
        id1 = save_config("keep",   {}, db_path=db)
        id2 = save_config("delete", {}, db_path=db)
        delete_config(id2, db_path=db)
        df  = list_configs(db_path=db)
        assert len(df) == 1
        assert df.iloc[0]["id"] == id1

    def test_deleting_nonexistent_does_not_raise(self, db):
        delete_config(99999, db_path=db)   # should not raise


class TestRenameConfig:
    def test_name_updated(self, db):
        rid = save_config("old_name", SAMPLE_CONFIG, db_path=db)
        rename_config(rid, "new_name", db_path=db)
        df = list_configs(db_path=db)
        assert df[df["id"] == rid]["name"].values[0] == "new_name"

    def test_config_data_unchanged_after_rename(self, db):
        rid = save_config("original", SAMPLE_CONFIG, db_path=db)
        rename_config(rid, "renamed", db_path=db)
        cfg = load_config(rid, db_path=db)
        assert cfg["ticker"] == "AAPL"

    def test_strips_whitespace(self, db):
        rid = save_config("test", {}, db_path=db)
        rename_config(rid, "  padded name  ", db_path=db)
        df  = list_configs(db_path=db)
        assert df.iloc[0]["name"] == "padded name"
