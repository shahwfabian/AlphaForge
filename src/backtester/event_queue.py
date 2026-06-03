"""
Centralized Event Queue for the AlphaForge backtesting engine.

Architecture
------------
All communication between engine components flows through this queue:

    MarketEvent  →  SignalHandler
    SignalEvent  →  PortfolioHandler
    OrderEvent   →  ExecutionHandler
    FillEvent    →  Portfolio + TradeLog

This decouples producers from consumers, mirrors how real trading systems
route events through a message bus (e.g. FIX engine, ITCH handler), and
makes the data-flow fully auditable.

Variants
--------
EventQueue        — synchronous (queue.Queue), used by BacktestEngine
AsyncEventQueue   — asyncio.Queue wrapper, used by the streaming layer
"""

import asyncio
import inspect
import logging
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .events import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger(__name__)

# Union type alias
AnyEvent = MarketEvent | SignalEvent | OrderEvent | FillEvent


# ── Queue statistics ──────────────────────────────────────────────────────────

@dataclass
class QueueStats:
    """Lightweight telemetry collected across a backtest run."""
    enqueued:    dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dequeued:    dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_enqueued: int = 0
    total_dequeued: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_put(self, event: AnyEvent) -> None:
        key = event.event_type.value
        self.enqueued[key] += 1
        self.total_enqueued += 1

    def record_get(self, event: AnyEvent) -> None:
        key = event.event_type.value
        self.dequeued[key] += 1
        self.total_dequeued += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def summary(self) -> dict:
        return {
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "enqueued_by_type": dict(self.enqueued),
            "dequeued_by_type": dict(self.dequeued),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


# ── Synchronous EventQueue ────────────────────────────────────────────────────

class EventQueue:
    """
    Thread-safe synchronous event queue backed by ``queue.Queue``.

    Handlers can be registered for each ``EventType``.  When ``dispatch()``
    is called the event is routed to every matching handler in registration
    order.  Unhandled event types are silently ignored (logged at DEBUG).

    Usage
    -----
    >>> eq = EventQueue()
    >>> eq.register(EventType.MARKET, my_handler)
    >>> eq.put(MarketEvent(...))
    >>> eq.dispatch_all()        # drains queue, fires handlers
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[AnyEvent] = queue.Queue()
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self.stats = QueueStats()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register(self, event_type: EventType, handler: Callable) -> None:
        """
        Register a callable to be invoked when ``event_type`` is dispatched.

        Multiple handlers per type are supported; they execute in FIFO order.
        """
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for %s.", handler.__name__, event_type)

    def unregister_all(self, event_type: EventType) -> None:
        """Remove all handlers for a given event type."""
        self._handlers.pop(event_type, None)

    # ------------------------------------------------------------------
    # Put / Get
    # ------------------------------------------------------------------

    def put(self, event: AnyEvent) -> None:
        """Enqueue an event."""
        self._queue.put_nowait(event)
        self.stats.record_put(event)

    def get(self) -> AnyEvent:
        """Dequeue one event (non-blocking)."""
        event = self._queue.get_nowait()
        self.stats.record_get(event)
        return event

    def empty(self) -> bool:
        """Return True if the queue contains no pending events."""
        return self._queue.empty()

    def size(self) -> int:
        """Approximate number of events currently in queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, event: AnyEvent) -> None:
        """
        Immediately route ``event`` to its registered handlers.

        Does NOT dequeue — caller should call ``get()`` first if the event
        was previously enqueued.
        """
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug("No handler for %s.", event.event_type)
            return
        for handler in handlers:
            handler(event)

    def dispatch_all(self) -> int:
        """
        Drain the queue and dispatch every pending event.

        Returns
        -------
        int
            Number of events dispatched.
        """
        dispatched = 0
        while not self.empty():
            event = self.get()
            self.dispatch(event)
            dispatched += 1
        return dispatched

    # ------------------------------------------------------------------
    # Context manager (auto-reset stats)
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventQueue":
        self.stats = QueueStats()
        return self

    def __exit__(self, *_) -> None:
        logger.debug("EventQueue session ended. %s", self.stats.summary())


# ── Asynchronous EventQueue ───────────────────────────────────────────────────

class AsyncEventQueue:
    """
    Async event queue backed by ``asyncio.Queue``.

    Designed for the real-time streaming layer where market data arrives
    over WebSockets and must be processed without blocking the event loop.

    Usage
    -----
    >>> aeq = AsyncEventQueue()
    >>> await aeq.put(MarketEvent(...))
    >>> async for event in aeq:
    ...     await handler(event)
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[AnyEvent] = asyncio.Queue(maxsize=maxsize)
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self.stats = QueueStats()

    def register(self, event_type: EventType, handler: Callable) -> None:
        """Register an async or sync handler for an event type."""
        self._handlers[event_type].append(handler)

    async def put(self, event: AnyEvent) -> None:
        await self._queue.put(event)
        self.stats.record_put(event)

    async def get(self) -> AnyEvent:
        event = await self._queue.get()
        self.stats.record_get(event)
        return event

    def empty(self) -> bool:
        return self._queue.empty()

    async def dispatch(self, event: AnyEvent) -> None:
        """Dispatch to all registered handlers (awaiting async ones)."""
        for handler in self._handlers.get(event.event_type, []):
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    async def dispatch_all(self) -> int:
        dispatched = 0
        while not self.empty():
            event = await self.get()
            await self.dispatch(event)
            dispatched += 1
        return dispatched

    def __aiter__(self):
        return self

    async def __anext__(self) -> AnyEvent:
        return await self.get()
