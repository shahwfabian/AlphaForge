"""
Event types for the AlphaForge event-driven backtesting engine.

Inspired by the classic event-driven backtester architecture described in
"Successful Algorithmic Trading" (Hurst, 2014).

Event hierarchy
---------------
MarketEvent  → new OHLCV bar has arrived
SignalEvent  → strategy has produced a directional signal
OrderEvent   → portfolio has decided to place an order
FillEvent    → broker has simulated execution of an order
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


# ---------------------------------------------------------------------------
# Market Event
# ---------------------------------------------------------------------------

@dataclass
class MarketEvent:
    """
    Fired once per bar as the engine advances through the data.

    Attributes
    ----------
    timestamp : datetime
        Date/time of the bar.
    ticker : str
        Asset symbol.
    open_ : float
    high : float
    low : float
    close : float
    volume : float
    """
    event_type: EventType = field(default=EventType.MARKET, init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    open_: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


# ---------------------------------------------------------------------------
# Signal Event
# ---------------------------------------------------------------------------

@dataclass
class SignalEvent:
    """
    Produced by a Strategy when it detects an entry or exit condition.

    Attributes
    ----------
    timestamp : datetime
    ticker : str
    signal : int
        +1 = long, 0 = flat / exit, -1 = short.
    strategy_name : str
    """
    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    signal: int = 0          # +1, 0, -1
    strategy_name: str = ""


# ---------------------------------------------------------------------------
# Order Event
# ---------------------------------------------------------------------------

@dataclass
class OrderEvent:
    """
    Instruction sent from the Portfolio to the execution handler.

    Attributes
    ----------
    timestamp : datetime
    ticker : str
    direction : OrderDirection
        BUY or SELL.
    quantity : float
        Number of shares / units.
    order_type : OrderType
        MARKET (default) or LIMIT.
    """
    event_type: EventType = field(default=EventType.ORDER, init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    direction: OrderDirection = OrderDirection.BUY
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET


# ---------------------------------------------------------------------------
# Fill Event
# ---------------------------------------------------------------------------

@dataclass
class FillEvent:
    """
    Represents the simulated execution of an OrderEvent.

    Attributes
    ----------
    timestamp : datetime
    ticker : str
    direction : OrderDirection
    quantity : float
        Actual shares filled.
    fill_price : float
        Execution price (includes slippage).
    commission : float
        Total transaction cost in dollar terms.
    """
    event_type: EventType = field(default=EventType.FILL, init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    direction: OrderDirection = OrderDirection.BUY
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0

    @property
    def fill_value(self) -> float:
        """Total notional value of the fill (excluding commission)."""
        return self.quantity * self.fill_price

    @property
    def net_cost(self) -> float:
        """Cash outflow (positive = bought, negative = sold + commission)."""
        sign = 1 if self.direction == OrderDirection.BUY else -1
        return sign * self.fill_value + self.commission
