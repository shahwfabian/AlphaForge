"""Tests for src/analytics/stress_testing.py"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.stress_testing import (
    run_stress_scenarios,
    custom_scenario,
    STRESS_SCENARIOS,
    _estimate_beta_proxy,
)


@pytest.fixture
def daily_returns():
    rng  = np.random.default_rng(42)
    n    = 252
    rets = pd.Series(rng.normal(0.0004, 0.012, n))
    return rets


class TestRunStressScenarios:
    def test_returns_dataframe(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        assert isinstance(df, pd.DataFrame)

    def test_not_empty(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        assert len(df) > 0

    def test_has_required_columns(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        for col in ("Severity_Rank", "Scenario", "Estimated_Loss_Pct",
                    "Estimated_Loss_USD", "Ending_Value"):
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_worst_to_best(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        losses = df["Estimated_Loss_Pct"].tolist()
        assert losses == sorted(losses)

    def test_severity_rank_starts_at_one(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        assert df["Severity_Rank"].iloc[0] == 1

    def test_all_predefined_scenarios_present(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        for name in STRESS_SCENARIOS:
            assert name in df["Scenario"].values, f"Missing: {name}"

    def test_losses_are_negative(self, daily_returns):
        df = run_stress_scenarios(daily_returns, initial_value=100_000)
        assert (df["Estimated_Loss_Pct"] < 0).all()

    def test_ending_value_less_than_initial(self, daily_returns):
        initial = 100_000
        df = run_stress_scenarios(daily_returns, initial_value=initial)
        assert (df["Ending_Value"] < initial).all()

    def test_custom_scenarios_accepted(self, daily_returns):
        custom = {"test scenario": {
            "description": "test",
            "market_shock_pct": -0.10,
            "vol_multiplier": 1.5,
            "duration_days": 21,
        }}
        df = run_stress_scenarios(daily_returns, initial_value=100_000, scenarios=custom)
        assert len(df) == 1
        assert df.iloc[0]["Scenario"] == "test scenario"


class TestCustomScenario:
    def test_returns_dict(self, daily_returns):
        result = custom_scenario(daily_returns, 100_000, -0.20)
        assert isinstance(result, dict)

    def test_has_required_keys(self, daily_returns):
        result = custom_scenario(daily_returns, 100_000, -0.20)
        for key in ("estimated_loss_pct", "estimated_loss_usd", "ending_value", "beta_proxy"):
            assert key in result

    def test_loss_is_negative_for_negative_shock(self, daily_returns):
        result = custom_scenario(daily_returns, 100_000, -0.20)
        assert result["estimated_loss_pct"] < 0
        assert result["estimated_loss_usd"] < 0

    def test_ending_value_less_than_initial(self, daily_returns):
        result = custom_scenario(daily_returns, 100_000, -0.30)
        assert result["ending_value"] < 100_000


class TestBetaProxy:
    def test_returns_float(self, daily_returns):
        b = _estimate_beta_proxy(daily_returns)
        assert isinstance(b, float)

    def test_clamped_to_max(self):
        very_volatile = pd.Series(np.random.normal(0, 0.10, 252))  # 100% ann vol
        b = _estimate_beta_proxy(very_volatile)
        assert b <= 2.5

    def test_clamped_to_min(self):
        flat = pd.Series([0.0] * 252)
        b = _estimate_beta_proxy(flat)
        assert b >= 0.05


class TestStressScenarioConstants:
    def test_all_scenarios_have_required_keys(self):
        for name, params in STRESS_SCENARIOS.items():
            assert "market_shock_pct" in params, f"{name} missing market_shock_pct"
            assert "duration_days" in params, f"{name} missing duration_days"
            assert params["market_shock_pct"] < 0, f"{name} shock should be negative"

    def test_scenario_count(self):
        assert len(STRESS_SCENARIOS) >= 5
