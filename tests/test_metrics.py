"""
Tests for performance, risk, and benchmark analytics.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.performance import (
    compute_all_metrics,
    cumulative_returns,
    drawdown_series,
)
from src.analytics.risk import (
    historical_var,
    parametric_var,
    expected_shortfall,
    rolling_volatility,
    compute_risk_metrics,
)
from src.analytics.benchmark import compute_benchmark_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def constant_return_equity() -> pd.DataFrame:
    """Equity curve with a constant 0.1% daily return — predictable metrics."""
    n = 252
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    daily_r = 0.001
    pv = 100_000 * (1 + daily_r) ** np.arange(n)
    ret = pd.Series([daily_r] * n, index=dates)
    df = pd.DataFrame(
        {"Portfolio_Value": pv, "Daily_Return": ret},
        index=dates,
    )
    df.loc[df.index[0], "Daily_Return"] = 0.0
    return df


@pytest.fixture
def negative_trend_equity() -> pd.DataFrame:
    """Equity curve that steadily declines — max drawdown should be large."""
    n = 252
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    daily_r = -0.002
    pv = 100_000 * (1 + daily_r) ** np.arange(n)
    ret = pd.Series([daily_r] * n, index=dates)
    df = pd.DataFrame(
        {"Portfolio_Value": pv, "Daily_Return": ret},
        index=dates,
    )
    return df


@pytest.fixture
def random_returns() -> pd.Series:
    """500 random daily returns ~ N(0.0003, 0.012)."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0003, 0.012, 500))


# ── Performance metrics ────────────────────────────────────────────────────────

class TestPerformanceMetrics:
    def test_cumulative_return_constant(self, constant_return_equity):
        metrics = compute_all_metrics(constant_return_equity)
        expected_cum = (1.001 ** 252) - 1
        assert metrics["Cumulative_Return"] == pytest.approx(expected_cum, rel=0.01)

    def test_annualised_return_positive(self, constant_return_equity):
        metrics = compute_all_metrics(constant_return_equity)
        assert metrics["Annualised_Return"] > 0

    def test_max_drawdown_is_zero_for_constant_up(self, constant_return_equity):
        """A monotonically increasing series should have zero drawdown."""
        metrics = compute_all_metrics(constant_return_equity)
        assert metrics["Max_Drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_negative_for_declining(self, negative_trend_equity):
        """Declining equity should have a large negative drawdown."""
        metrics = compute_all_metrics(negative_trend_equity)
        assert metrics["Max_Drawdown"] < -0.30

    def test_sharpe_positive_for_constant_up(self, constant_return_equity):
        """Constant positive return should yield a positive Sharpe ratio."""
        metrics = compute_all_metrics(constant_return_equity)
        assert metrics["Sharpe_Ratio"] > 0

    def test_sharpe_negative_for_declining(self, negative_trend_equity):
        metrics = compute_all_metrics(negative_trend_equity)
        assert metrics["Sharpe_Ratio"] < 0

    def test_cumulative_returns_helper(self):
        rets = pd.Series([0.01, -0.01, 0.02])
        cum = cumulative_returns(rets)
        # (1.01)(0.99)(1.02) - 1
        expected = (1.01 * 0.99 * 1.02) - 1
        assert cum.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_drawdown_series_never_positive(self, random_returns):
        """Drawdown series should be <= 0 by definition."""
        eq = (1 + random_returns).cumprod() * 100_000
        dd = drawdown_series(random_returns)
        assert (dd <= 1e-9).all()

    def test_win_rate_between_zero_and_one(self, constant_return_equity):
        metrics = compute_all_metrics(constant_return_equity)
        assert 0.0 <= metrics["Win_Rate"] <= 1.0


# ── Risk metrics ───────────────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_var_is_negative(self, random_returns):
        """Historical VaR should be a negative number (loss)."""
        var = historical_var(random_returns, confidence=0.95)
        assert var < 0

    def test_var_99_more_extreme_than_95(self, random_returns):
        """99% VaR loss should be larger in magnitude than 95% VaR."""
        var95 = historical_var(random_returns, confidence=0.95)
        var99 = historical_var(random_returns, confidence=0.99)
        assert var99 < var95  # Both negative; 99 is more negative

    def test_expected_shortfall_more_extreme_than_var(self, random_returns):
        """ES should be at least as extreme (negative) as VaR."""
        var = historical_var(random_returns, confidence=0.95)
        es = expected_shortfall(random_returns, confidence=0.95)
        assert es <= var

    def test_parametric_var_similar_to_historical(self, random_returns):
        """For normal-like returns the two VaR methods should be close."""
        h_var = historical_var(random_returns, confidence=0.95)
        p_var = parametric_var(random_returns, confidence=0.95)
        # Within 50% of each other for synthetic normal data
        assert abs(h_var - p_var) / abs(h_var) < 0.5

    def test_rolling_vol_length(self, random_returns):
        roll_vol = rolling_volatility(random_returns, window=21)
        assert len(roll_vol) == len(random_returns)

    def test_risk_metrics_dict_keys(self, random_returns):
        metrics = compute_risk_metrics(random_returns)
        required = {"VaR_Historical_95", "Expected_Shortfall_95", "Skewness", "Kurtosis"}
        assert required.issubset(metrics.keys())

    def test_empty_returns_returns_empty_dict(self):
        assert compute_risk_metrics(pd.Series(dtype=float)) == {}


# ── Benchmark metrics ──────────────────────────────────────────────────────────

class TestBenchmarkMetrics:
    @pytest.fixture
    def returns_pair(self):
        rng = np.random.default_rng(99)
        bm = pd.Series(rng.normal(0.0004, 0.010, 252))
        strat = bm * 0.8 + rng.normal(0, 0.003, 252)  # ~correlated to bm
        return strat, bm

    def test_beta_close_to_0point8(self, returns_pair):
        strat, bm = returns_pair
        metrics = compute_benchmark_metrics(strat, bm)
        assert metrics["Beta"] == pytest.approx(0.8, abs=0.15)

    def test_correlation_high(self, returns_pair):
        strat, bm = returns_pair
        metrics = compute_benchmark_metrics(strat, bm)
        assert metrics["Correlation"] > 0.7

    def test_perfect_correlation(self):
        rets = pd.Series(np.random.normal(0.001, 0.01, 300))
        metrics = compute_benchmark_metrics(rets, rets)
        assert metrics["Correlation"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["Beta"] == pytest.approx(1.0, abs=1e-6)

    def test_required_keys_present(self, returns_pair):
        strat, bm = returns_pair
        metrics = compute_benchmark_metrics(strat, bm)
        for key in ["Alpha", "Beta", "Correlation", "Tracking_Error", "Information_Ratio"]:
            assert key in metrics

    def test_insufficient_data_returns_empty(self):
        s = pd.Series([0.01, 0.02, 0.03])
        b = pd.Series([0.01, 0.02, 0.03])
        result = compute_benchmark_metrics(s, b)
        assert result == {}
