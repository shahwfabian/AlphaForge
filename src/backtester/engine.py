"""
AlphaForge Event-Driven Backtesting Engine — Queue Architecture.

Data-flow (per bar)
-------------------
    put(MarketEvent)
        └─ _on_market()    reads pre-computed signal
               └─ put(SignalEvent)   if signal changed
                      └─ _on_signal()   determines order direction
                             └─ put(OrderEvent)
                                    └─ _on_order()   simulates fill
                                           └─ put(FillEvent)
                                                  └─ _on_fill()   updates portfolio

All communication flows through the centralized EventQueue.
Handlers are registered at construction time — adding new handlers (e.g.
a risk-limit checker between OrderEvent and FillEvent) requires zero
changes to existing code.

Lookahead bias
--------------
Strategy signals are pre-computed using BaseStrategy._finalise(), which
shifts the position series by 1 bar.  The engine then reads the already-
shifted Position column — no future data is ever visible during simulation.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .event_queue import EventQueue
from .events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalEvent,
)
from .portfolio import Portfolio
from .trade_log import TradeLog
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Queue-based event-driven backtesting engine.

    Parameters
    ----------
    strategy : BaseStrategy
        Instantiated strategy object.
    initial_capital : float
        Starting portfolio value in USD.
    transaction_cost : float
        Commission rate (fraction of notional, e.g. 0.001 = 0.1 %).
    slippage : float
        One-way slippage rate (fraction of price, e.g. 0.0005 = 0.05 %).
    vol_target : float or None
        Annualised volatility target for position sizing.  None = 95 % cash.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        vol_target: Optional[float] = None,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.vol_target = vol_target

        self.portfolio = Portfolio(
            initial_capital, transaction_cost, slippage, vol_target
        )
        self.trade_log = TradeLog()
        self.event_queue = EventQueue()

        # Mutable state threaded through event handlers
        self._signals_df: Optional[pd.DataFrame] = None
        self._current_bar: Optional[pd.Series] = None
        self._current_ts: Optional[datetime] = None
        self._current_ticker: str = ""
        self._prev_position: int = 0

        # Wire handlers into the queue
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration (the plugin attachment point)
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """
        Register event handlers.

        Adding a new cross-cutting concern (e.g. a risk limit that caps
        order size, or a latency model) requires only adding a new handler
        here — no changes to existing handlers.
        """
        self.event_queue.register(EventType.MARKET, self._on_market)
        self.event_queue.register(EventType.SIGNAL, self._on_signal)
        self.event_queue.register(EventType.ORDER,  self._on_order)
        self.event_queue.register(EventType.FILL,   self._on_fill)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Execute the full backtest on a preprocessed OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-processed data with 'Close' and 'Return' columns, date index.
        ticker : str
            Asset symbol.

        Returns
        -------
        dict
            signals, equity_curve, trade_log, strategy, ticker.
        """
        logger.info(
            "Backtest — strategy: %s | asset: %s | bars: %d | vol_target: %s",
            self.strategy.name, ticker, len(df),
            f"{self.vol_target:.0%}" if self.vol_target else "fixed",
        )

        self._current_ticker = ticker

        # Pre-compute all signals (vectorised; no lookahead)
        self._signals_df = self.strategy.generate_signals(df)

        # Run the event loop
        with self.event_queue:
            self._run_event_loop(df, ticker)

        equity_df = self.portfolio.get_equity_df()
        trade_df  = self.trade_log.to_dataframe()
        equity_df = self._attach_returns(equity_df, self._signals_df)

        final_equity = (
            equity_df["Portfolio_Value"].iloc[-1]
            if not equity_df.empty else self.initial_capital
        )
        logger.info(
            "Backtest complete | trades: %d | final equity: $%.2f | "
            "queue stats: %s",
            self.trade_log.count_trades(),
            final_equity,
            self.event_queue.stats.summary(),
        )

        return {
            "signals":      self._signals_df,
            "equity_curve": equity_df,
            "trade_log":    trade_df,
            "strategy":     self.strategy.name,
            "ticker":       ticker,
            "queue_stats":  self.event_queue.stats.summary(),
        }

    # ------------------------------------------------------------------
    # Event loop — one MarketEvent per bar, then drain queue
    # ------------------------------------------------------------------

    def _run_event_loop(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Advance through the data bar by bar.

        For each bar:
          1. Store bar context so handlers can read it.
          2. Put a MarketEvent on the queue.
          3. Drain the queue — handlers chain further events as needed.
          4. Record the equity snapshot.
        """
        for timestamp, row in df.iterrows():
            # Store bar context (shared mutable state for this tick)
            self._current_ts  = timestamp
            self._current_bar = row

            # Fire the root event
            self.event_queue.put(
                MarketEvent(
                    timestamp=timestamp,
                    ticker=ticker,
                    open_=float(row.get("Open",  row["Close"])),
                    high= float(row.get("High",  row["Close"])),
                    low=  float(row.get("Low",   row["Close"])),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )

            # Drain — every handler may enqueue further events
            self.event_queue.dispatch_all()

            # Daily equity snapshot (after all fills settle)
            self.portfolio.record_equity(
                timestamp, {ticker: float(row["Close"])}
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_market(self, event: MarketEvent) -> None:
        """
        Handle MarketEvent: read the pre-computed signal and emit
        a SignalEvent if the desired position has changed.
        """
        if self._signals_df is None:
            return
        if event.timestamp not in self._signals_df.index:
            return

        row = self._signals_df.loc[event.timestamp]
        current_signal:   int = int(row.get("Signal",   0))
        current_position: int = int(row.get("Position", 0))

        if current_position != self._prev_position:
            self.event_queue.put(
                SignalEvent(
                    timestamp=event.timestamp,
                    ticker=event.ticker,
                    signal=current_signal,
                    strategy_name=self.strategy.name,
                )
            )
        # Store position for next bar comparison
        self._prev_position = current_position

    def _on_signal(self, event: SignalEvent) -> None:
        """
        Handle SignalEvent: determine order direction and emit an OrderEvent.
        """
        close = float(self._current_bar["Close"])
        held  = self.portfolio.get_position(event.ticker)

        if event.signal > 0 and held == 0:
            # Going long
            fraction = self.portfolio.compute_position_fraction()
            quantity = max(1.0, round(self.portfolio.cash * fraction / close, 2))
            self.event_queue.put(
                OrderEvent(
                    timestamp=event.timestamp,
                    ticker=event.ticker,
                    direction=OrderDirection.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )
            )
        elif event.signal == 0 and held > 0:
            # Exiting
            self.event_queue.put(
                OrderEvent(
                    timestamp=event.timestamp,
                    ticker=event.ticker,
                    direction=OrderDirection.SELL,
                    quantity=held,
                    order_type=OrderType.MARKET,
                )
            )

    def _on_order(self, event: OrderEvent) -> None:
        """
        Handle OrderEvent: simulate fill with slippage + commission
        and emit a FillEvent.
        """
        close      = float(self._current_bar["Close"])
        fill_price = self.portfolio.apply_slippage(close, event.direction)
        commission = self.portfolio.calculate_commission(event.quantity * fill_price)

        self.event_queue.put(
            FillEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction=event.direction,
                quantity=event.quantity,
                fill_price=fill_price,
                commission=commission,
            )
        )

    def _on_fill(self, event: FillEvent) -> None:
        """
        Handle FillEvent: update portfolio cash/holdings and record trade.
        """
        self.portfolio.process_fill(event)
        pv = self.portfolio.total_equity(
            {event.ticker: float(self._current_bar["Close"])}
        )
        # Read the signal column for the current bar
        sig = 0
        if self._signals_df is not None and self._current_ts in self._signals_df.index:
            sig = int(self._signals_df.loc[self._current_ts].get("Signal", 0))
        self.trade_log.record(event, sig, pv)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_returns(
        equity_df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if equity_df.empty or signals_df.empty:
            return equity_df

        equity_df["Daily_Return"] = equity_df["Portfolio_Value"].pct_change()

        if "Strategy_Return" in signals_df.columns:
            equity_df = equity_df.join(
                signals_df[["Strategy_Return", "Asset_Return"]].rename(
                    columns={
                        "Strategy_Return": "Strat_Return",
                        "Asset_Return":    "Benchmark_Return",
                    }
                ),
                how="left",
            )

        equity_df["Cumulative_Return"] = (
            equity_df["Portfolio_Value"] / equity_df["Portfolio_Value"].iloc[0] - 1
        )
        return equity_df
