"""
Tests for the backtesting engine and portfolio simulation.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestEngine
from src.backtester.portfolio import Portfolio
from src.backtester.events import FillEvent, OrderDirection
from src.strategies.buy_hold import BuyAndHold
from src.strategies.moving_average import MovingAverageCrossover


INITIAL_CAPITAL = 100_000.0


# ── Portfolio tests ────────────────────────────────────────────────────────────

class TestPortfolio:
    def test_initial_cash(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL)
        assert p.cash == INITIAL_CAPITAL

    def test_buy_reduces_cash(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, transaction_cost=0.0, slippage=0.0)
        fill = FillEvent(
            ticker="TEST",
            direction=OrderDirection.BUY,
            quantity=10.0,
            fill_price=100.0,
            commission=0.0,
        )
        p.process_fill(fill)
        assert p.cash == INITIAL_CAPITAL - 1000.0
        assert p.get_position("TEST") == 10.0

    def test_sell_increases_cash(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, transaction_cost=0.0, slippage=0.0)
        # First buy 10 shares
        buy = FillEvent(ticker="TEST", direction=OrderDirection.BUY,
                        quantity=10.0, fill_price=100.0, commission=0.0)
        p.process_fill(buy)
        # Then sell all
        sell = FillEvent(ticker="TEST", direction=OrderDirection.SELL,
                         quantity=10.0, fill_price=110.0, commission=0.0)
        p.process_fill(sell)
        assert p.get_position("TEST") == 0.0
        assert p.cash == pytest.approx(INITIAL_CAPITAL + 100.0, rel=1e-6)

    def test_commission_reduces_proceeds(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, transaction_cost=0.001, slippage=0.0)
        fill = FillEvent(ticker="TEST", direction=OrderDirection.BUY,
                         quantity=100.0, fill_price=100.0, commission=10.0)
        p.process_fill(fill)
        expected_cash = INITIAL_CAPITAL - 100.0 * 100.0 - 10.0
        assert p.cash == pytest.approx(expected_cash, rel=1e-6)

    def test_cannot_go_negative_cash(self):
        """Buying more than available cash should be capped."""
        p = Portfolio(initial_capital=1_000.0, transaction_cost=0.0, slippage=0.0)
        fill = FillEvent(ticker="TEST", direction=OrderDirection.BUY,
                         quantity=100.0, fill_price=100.0, commission=0.0)
        p.process_fill(fill)
        # Cash should not be negative
        assert p.cash >= 0.0

    def test_total_equity_with_holdings(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, transaction_cost=0.0, slippage=0.0)
        fill = FillEvent(ticker="TEST", direction=OrderDirection.BUY,
                         quantity=500.0, fill_price=100.0, commission=0.0)
        p.process_fill(fill)
        equity = p.total_equity({"TEST": 120.0})
        # Cash: 50000, Holdings: 500 * 120 = 60000, Total: 110000
        assert equity == pytest.approx(INITIAL_CAPITAL + 500 * 20.0, rel=1e-6)

    def test_slippage_applied_to_buy(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, slippage=0.01)
        slipped_price = p.apply_slippage(100.0, OrderDirection.BUY)
        assert slipped_price == pytest.approx(101.0)

    def test_slippage_applied_to_sell(self):
        p = Portfolio(initial_capital=INITIAL_CAPITAL, slippage=0.01)
        slipped_price = p.apply_slippage(100.0, OrderDirection.SELL)
        assert slipped_price == pytest.approx(99.0)


# ── BacktestEngine tests ───────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_run_returns_required_keys(self, processed_ohlcv):
        engine = BacktestEngine(BuyAndHold(), initial_capital=INITIAL_CAPITAL)
        result = engine.run(processed_ohlcv, ticker="TEST")
        assert "equity_curve" in result
        assert "trade_log" in result
        assert "signals" in result
        assert "strategy" in result

    def test_equity_curve_not_empty(self, processed_ohlcv):
        engine = BacktestEngine(BuyAndHold(), initial_capital=INITIAL_CAPITAL)
        result = engine.run(processed_ohlcv, ticker="TEST")
        assert not result["equity_curve"].empty

    def test_initial_capital_preserved_at_start(self, processed_ohlcv):
        """First equity row should be near initial capital."""
        engine = BacktestEngine(BuyAndHold(), initial_capital=INITIAL_CAPITAL,
                                transaction_cost=0.0, slippage=0.0)
        result = engine.run(processed_ohlcv, ticker="TEST")
        eq = result["equity_curve"]["Portfolio_Value"]
        # First value should equal initial capital (nothing bought yet on day 0)
        assert eq.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)

    def test_transaction_costs_reduce_returns(self, processed_ohlcv):
        """Portfolio with non-zero transaction costs should end lower than zero-cost."""
        strat = MovingAverageCrossover(short_window=10, long_window=30)

        engine_free = BacktestEngine(
            MovingAverageCrossover(short_window=10, long_window=30),
            initial_capital=INITIAL_CAPITAL, transaction_cost=0.0, slippage=0.0,
        )
        engine_paid = BacktestEngine(
            MovingAverageCrossover(short_window=10, long_window=30),
            initial_capital=INITIAL_CAPITAL, transaction_cost=0.005, slippage=0.001,
        )

        res_free = engine_free.run(processed_ohlcv, "TEST")
        res_paid = engine_paid.run(processed_ohlcv, "TEST")

        free_end = res_free["equity_curve"]["Portfolio_Value"].iloc[-1]
        paid_end = res_paid["equity_curve"]["Portfolio_Value"].iloc[-1]

        # Costs should reduce ending wealth (assuming there are trades)
        # Allow for the degenerate case where no trades are made
        trade_count = res_paid["trade_log"][
            res_paid["trade_log"]["Direction"].isin(["BUY", "SELL"])
        ].shape[0]
        if trade_count > 0:
            assert paid_end <= free_end

    def test_equity_curve_has_correct_length(self, processed_ohlcv):
        """Equity curve should have approximately the same rows as input data."""
        engine = BacktestEngine(BuyAndHold(), initial_capital=INITIAL_CAPITAL)
        result = engine.run(processed_ohlcv, ticker="TEST")
        # Should be within a few rows of the input length
        assert abs(len(result["equity_curve"]) - len(processed_ohlcv)) <= 5

    def test_no_future_data_used(self, processed_ohlcv):
        """
        Verify no lookahead bias: strategy returns on day t should only
        use information available up to day t-1.
        We check this by confirming Position = Signal.shift(1).
        """
        engine = BacktestEngine(
            MovingAverageCrossover(short_window=10, long_window=30),
            initial_capital=INITIAL_CAPITAL,
        )
        result = engine.run(processed_ohlcv, "TEST")
        signals = result["signals"]
        expected_pos = signals["Signal"].shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(
            signals["Position"], expected_pos, check_names=False
        )
