"""
Abstract base interfaces for real-time market data streaming.

Design principles
-----------------
- Protocol-agnostic: HTTP, WebSocket, FIX, or file replay all implement
  the same ``MarketDataStream`` interface.
- Async-first: all I/O methods are coroutines compatible with asyncio.
- Stateful: streams expose a ``status`` property for health monitoring.
- Observable: implementations should emit ``MarketEvent`` objects that
  slot directly into ``AsyncEventQueue``.

Implementing a new data source
-------------------------------
1. Subclass ``MarketDataStream``.
2. Implement ``connect()``, ``subscribe()``, ``stream()``, ``disconnect()``.
3. Register the class in ``src/streaming/__init__.py``.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator

from ..backtester.events import MarketEvent

logger = logging.getLogger(__name__)


# ── Stream status ─────────────────────────────────────────────────────────────

class StreamStatus(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING   = "CONNECTING"
    CONNECTED    = "CONNECTED"
    SUBSCRIBED   = "SUBSCRIBED"
    STREAMING    = "STREAMING"
    ERROR        = "ERROR"
    CLOSED       = "CLOSED"


# ── Stream configuration ──────────────────────────────────────────────────────

@dataclass
class StreamConfig:
    """
    Configuration for a market data stream connection.

    Parameters
    ----------
    uri : str
        WebSocket URI or HTTP endpoint.
    tickers : list[str]
        Symbols to subscribe to.
    reconnect_attempts : int
        Maximum automatic reconnection attempts on disconnect.
    reconnect_delay_s : float
        Seconds to wait between reconnection attempts.
    heartbeat_interval_s : float
        How often to send a heartbeat / ping (0 = disabled).
    max_queue_size : int
        Maximum number of events to buffer in the async queue.
    """
    uri:                    str
    tickers:                list[str] = field(default_factory=list)
    reconnect_attempts:     int   = 3
    reconnect_delay_s:      float = 2.0
    heartbeat_interval_s:   float = 30.0
    max_queue_size:         int   = 10_000


# ── Abstract stream interface ─────────────────────────────────────────────────

class MarketDataStream(ABC):
    """
    Abstract base class for all market data stream providers.

    Subclasses must implement:
        connect()    — establish connection
        subscribe()  — subscribe to tickers
        stream()     — async generator yielding MarketEvent objects
        disconnect() — gracefully close connection

    Thread-safety: all methods are async-safe for use with asyncio.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config  = config
        self._status = StreamStatus.DISCONNECTED
        self._subscribed_tickers: list[str] = []
        self._event_count: int = 0
        self._connected_at: datetime | None = None

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish a connection to the data source.

        Should update ``self._status`` to ``CONNECTED`` on success.
        Raise ``ConnectionError`` on failure.
        """

    @abstractmethod
    async def subscribe(self, tickers: list[str]) -> None:
        """
        Subscribe to a list of ticker symbols.

        Should update ``self._status`` to ``SUBSCRIBED`` and populate
        ``self._subscribed_tickers``.
        """

    @abstractmethod
    async def stream(self) -> AsyncIterator[MarketEvent]:
        """
        Async generator that yields ``MarketEvent`` objects.

        Should update ``self._status`` to ``STREAMING`` on first event.
        Should handle reconnection logic internally.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully close the connection and clean up resources.

        Should update ``self._status`` to ``CLOSED``.
        """

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MarketDataStream":
        await self.connect()
        if self.config and self.config.tickers:
            await self.subscribe(self.config.tickers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Status and telemetry
    # ------------------------------------------------------------------

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status in (
            StreamStatus.CONNECTED,
            StreamStatus.SUBSCRIBED,
            StreamStatus.STREAMING,
        )

    @property
    def event_count(self) -> int:
        """Total MarketEvents emitted since connect()."""
        return self._event_count

    def uptime_seconds(self) -> float | None:
        """Seconds since connect() was called."""
        if self._connected_at is None:
            return None
        return (datetime.now(timezone.utc) - self._connected_at).total_seconds()

    def stats(self) -> dict:
        return {
            "status":       self._status.value,
            "tickers":      self._subscribed_tickers,
            "event_count":  self._event_count,
            "uptime_s":     self.uptime_seconds(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, status: StreamStatus) -> None:
        logger.debug("Stream status: %s → %s", self._status.value, status.value)
        self._status = status

    def _increment_event_count(self) -> None:
        self._event_count += 1
