"""
Portfolio simulation module.

Tracks cash, positions, and equity curve while processing FillEvents.
Applies transaction costs (commission) and slippage to every fill.
Supports optional volatility-targeting position sizing.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .events import FillEvent, OrderDirection

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Simulates a single-asset long-only portfolio with optional volatility targeting.

    Volatility Targeting
    --------------------
    When ``vol_target`` is set (e.g. 0.15 for 15 % annualised), the fraction
    of capital deployed on each entry is scaled by:

        fraction = vol_target / realised_vol

    where realised_vol = rolling std of recent daily returns × sqrt(252).
    The fraction is capped at 1.0 (no leverage) and floored at 0.1.

    This is inspired by risk-parity and CTA position sizing techniques.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance in USD.
    transaction_cost : float
        Commission as a fraction of trade notional (e.g. 0.001 = 0.1 %).
    slippage : float
        One-way slippage as a fraction of price (e.g. 0.0005 = 0.05 %).
    vol_target : float or None
        Target annualised portfolio volatility.  None = size at 95 % cash.
    vol_lookback : int
        Days used to estimate realised volatility for sizing (default 21).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        vol_target: Optional[float] = None,
        vol_lookback: int = 21,
    ) -> None:
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

        # Current state
        self.cash: float = initial_capital
        self.holdings: dict[str, float] = {}   # ticker → shares held
        self.avg_cost: dict[str, float] = {}   # ticker → avg cost basis

        # History
        self.equity_curve: list[dict] = []
        self.fill_history: list[FillEvent] = []
        self._daily_returns: list[float] = []  # Used for vol targeting

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_fraction(self) -> float:
        """
        Determine what fraction of cash to deploy on the next entry.

        If vol_target is None → return 0.95 (fixed sizing).
        Otherwise → scale by vol_target / realised_vol, clamped to [0.10, 1.00].
        """
        if self.vol_target is None or len(self._daily_returns) < self.vol_lookback:
            return 0.95

        recent = pd.Series(self._daily_returns[-self.vol_lookback:])
        realised_vol = recent.std() * np.sqrt(252)

        if realised_vol <= 0:
            return 0.95

        fraction = self.vol_target / realised_vol
        return float(np.clip(fraction, 0.10, 1.00))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> float:
        """Return current share count for ticker (0 if not held)."""
        return self.holdings.get(ticker, 0.0)

    def market_value(self, ticker: str, price: float) -> float:
        """Current market value of holdings in a ticker."""
        return self.get_position(ticker) * price

    def total_equity(self, prices: dict[str, float]) -> float:
        """Cash + sum of all holdings at current prices."""
        holdings_value = sum(
            qty * prices.get(tkr, 0.0) for tkr, qty in self.holdings.items()
        )
        return self.cash + holdings_value

    def unrealised_pnl(self, ticker: str, current_price: float) -> float:
        """Unrealised P&L on an open position."""
        qty = self.get_position(ticker)
        if qty <= 0:
            return 0.0
        cost_basis = self.avg_cost.get(ticker, current_price)
        return qty * (current_price - cost_basis)

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def apply_slippage(self, price: float, direction: OrderDirection) -> float:
        """Adjust fill price for market impact / slippage."""
        if direction == OrderDirection.BUY:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def calculate_commission(self, fill_value: float) -> float:
        """Commission = transaction_cost_rate × notional value."""
        return abs(fill_value) * self.transaction_cost

    def process_fill(self, fill: FillEvent) -> None:
        """
        Update cash and holdings after a simulated fill.

        Parameters
        ----------
        fill : FillEvent
            The completed order to record.
        """
        ticker = fill.ticker
        qty = fill.quantity
        price = fill.fill_price
        commission = fill.commission
        direction = fill.direction

        if direction == OrderDirection.BUY:
            cost = qty * price + commission
            if cost > self.cash:
                qty = max(0, (self.cash - commission) / (price if price > 0 else 1))
                cost = qty * price + commission
                logger.warning(
                    "Insufficient cash. Order for %s reduced to %.2f shares.", ticker, qty
                )
            if qty <= 0:
                return

            self.cash -= cost
            prev_qty = self.holdings.get(ticker, 0.0)
            prev_cost = self.avg_cost.get(ticker, 0.0)
            new_qty = prev_qty + qty
            self.avg_cost[ticker] = (
                (prev_qty * prev_cost + qty * price) / new_qty if new_qty > 0 else 0.0
            )
            self.holdings[ticker] = new_qty

        else:  # SELL
            held = self.holdings.get(ticker, 0.0)
            qty = min(qty, held)
            if qty <= 0:
                return

            proceeds = qty * price - commission
            self.cash += proceeds
            self.holdings[ticker] = held - qty
            if self.holdings[ticker] <= 0:
                self.holdings.pop(ticker, None)
                self.avg_cost.pop(ticker, None)

        self.fill_history.append(fill)

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def record_equity(self, timestamp: datetime, prices: dict[str, float]) -> None:
        """Append one row to the equity curve."""
        equity = self.total_equity(prices)

        # Track daily return for vol targeting
        if self.equity_curve:
            prev_equity = self.equity_curve[-1]["Portfolio_Value"]
            if prev_equity > 0:
                self._daily_returns.append(equity / prev_equity - 1)

        self.equity_curve.append(
            {
                "Date": timestamp,
                "Cash": self.cash,
                "Holdings_Value": equity - self.cash,
                "Portfolio_Value": equity,
            }
        )

    def get_equity_df(self) -> pd.DataFrame:
        """Return equity curve as a clean DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df.set_index("Date", inplace=True)
        return df
