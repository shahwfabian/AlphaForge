"""
Tests for the centralized EventQueue and AsyncEventQueue.
"""

import asyncio
import pytest
from datetime import datetime, timezone

from src.backtester.event_queue import EventQueue, AsyncEventQueue
from src.backtester.events import (
    EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent, OrderDirection
)


def _make_market_event() -> MarketEvent:
    return MarketEvent(
        timestamp=datetime.now(timezone.utc),
        ticker="AAPL",
        open_=150.0, high=152.0, low=149.0, close=151.0, volume=1_000_000.0,
    )

def _make_signal_event(signal: int = 1) -> SignalEvent:
    return SignalEvent(
        timestamp=datetime.now(timezone.utc),
        ticker="AAPL",
        signal=signal,
        strategy_name="Test",
    )

def _make_order_event() -> OrderEvent:
    return OrderEvent(
        timestamp=datetime.now(timezone.utc),
        ticker="AAPL",
        direction=OrderDirection.BUY,
        quantity=100.0,
    )

def _make_fill_event() -> FillEvent:
    return FillEvent(
        timestamp=datetime.now(timezone.utc),
        ticker="AAPL",
        direction=OrderDirection.BUY,
        quantity=100.0,
        fill_price=151.5,
        commission=1.51,
    )


class TestEventQueue:
    def test_put_and_get(self):
        q = EventQueue()
        event = _make_market_event()
        q.put(event)
        assert not q.empty()
        retrieved = q.get()
        assert retrieved is event
        assert q.empty()

    def test_fifo_order(self):
        q = EventQueue()
        events = [_make_market_event(), _make_signal_event(), _make_order_event()]
        for e in events:
            q.put(e)
        retrieved = [q.get() for _ in events]
        assert [r.event_type for r in retrieved] == [e.event_type for e in events]

    def test_empty_when_no_events(self):
        q = EventQueue()
        assert q.empty()
        assert q.size() == 0

    def test_size_tracks_queue_length(self):
        q = EventQueue()
        for _ in range(5):
            q.put(_make_market_event())
        assert q.size() == 5

    def test_handler_registration_and_dispatch(self):
        q = EventQueue()
        received = []
        q.register(EventType.MARKET, lambda e: received.append(e))

        event = _make_market_event()
        q.put(event)
        q.dispatch_all()

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_handlers_same_type(self):
        q = EventQueue()
        log1, log2 = [], []
        q.register(EventType.SIGNAL, lambda e: log1.append(1))
        q.register(EventType.SIGNAL, lambda e: log2.append(2))

        q.put(_make_signal_event())
        q.dispatch_all()

        assert log1 == [1]
        assert log2 == [2]

    def test_dispatch_all_returns_count(self):
        q = EventQueue()
        for _ in range(7):
            q.put(_make_market_event())
        dispatched = q.dispatch_all()
        assert dispatched == 7
        assert q.empty()

    def test_handler_not_called_for_wrong_type(self):
        q = EventQueue()
        received = []
        q.register(EventType.FILL, lambda e: received.append(e))

        q.put(_make_market_event())   # MARKET event
        q.dispatch_all()

        assert received == []         # FILL handler not triggered

    def test_stats_tracking(self):
        q = EventQueue()
        q.register(EventType.MARKET, lambda e: None)
        for _ in range(3):
            q.put(_make_market_event())
        q.dispatch_all()

        stats = q.stats.summary()
        assert stats["total_enqueued"] == 3
        assert stats["total_dequeued"] == 3

    def test_context_manager_resets_stats(self):
        q = EventQueue()
        q.put(_make_market_event())
        q.get()

        with q:
            assert q.stats.total_enqueued == 0

    def test_unregister_all(self):
        q = EventQueue()
        received = []
        q.register(EventType.MARKET, lambda e: received.append(e))
        q.unregister_all(EventType.MARKET)

        q.put(_make_market_event())
        q.dispatch_all()
        assert received == []

    def test_chained_event_production(self):
        """Handler can put new events — dispatch_all should process them."""
        q = EventQueue()
        order_received = []

        def on_signal(event):
            # Signal handler puts an order event
            q.put(_make_order_event())

        def on_order(event):
            order_received.append(event)

        q.register(EventType.SIGNAL, on_signal)
        q.register(EventType.ORDER,  on_order)

        q.put(_make_signal_event())
        q.dispatch_all()  # Should process signal → order within same call

        assert len(order_received) == 1


class TestAsyncEventQueue:
    def test_async_put_and_get(self):
        async def _run():
            q = AsyncEventQueue()
            event = _make_market_event()
            await q.put(event)
            retrieved = await q.get()
            assert retrieved is event
        asyncio.run(_run())

    def test_async_dispatch_all(self):
        async def _run():
            q = AsyncEventQueue()
            received = []
            q.register(EventType.MARKET, lambda e: received.append(e))
            await q.put(_make_market_event())
            count = await q.dispatch_all()
            assert count == 1
            assert len(received) == 1
        asyncio.run(_run())

    def test_async_empty(self):
        async def _run():
            q = AsyncEventQueue()
            assert q.empty()
            await q.put(_make_market_event())
            assert not q.empty()
        asyncio.run(_run())

    def test_async_handler_coroutine(self):
        async def _run():
            q = AsyncEventQueue()
            results = []

            async def async_handler(e):
                await asyncio.sleep(0)
                results.append(e.event_type)

            q.register(EventType.SIGNAL, async_handler)
            await q.put(_make_signal_event())
            await q.dispatch_all()
            assert results == [EventType.SIGNAL]
        asyncio.run(_run())
