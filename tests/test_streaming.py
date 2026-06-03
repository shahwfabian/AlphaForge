"""
Tests for the streaming interfaces and mock stream.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd

from src.streaming.interfaces import StreamStatus, StreamConfig
from src.streaming.mock_stream import MockMarketStream
from src.backtester.events import MarketEvent


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """Tiny OHLCV DataFrame for fast streaming tests."""
    n = 10
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": [1_000_000.0] * n,
    }, index=dates)


class TestMockMarketStream:
    def test_initial_status_is_disconnected(self, small_ohlcv):
        stream = MockMarketStream(small_ohlcv, ticker="TEST")
        assert stream.status == StreamStatus.DISCONNECTED

    def test_connect_sets_connected(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="TEST")
            await stream.connect()
            assert stream.status == StreamStatus.CONNECTED
        asyncio.run(_run())

    def test_stream_yields_market_events(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="TEST")
            events = await stream.collect_all()
            return events
        events = asyncio.run(_run())
        assert len(events) == len(small_ohlcv)
        assert all(isinstance(e, MarketEvent) for e in events)

    def test_event_ticker_correct(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="AAPL")
            return await stream.collect_all()
        events = asyncio.run(_run())
        assert all(e.ticker == "AAPL" for e in events)

    def test_event_close_prices_match(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="T")
            return await stream.collect_all()
        events = asyncio.run(_run())
        actual_closes   = [e.close for e in events]
        expected_closes = small_ohlcv["Close"].tolist()
        for a, b in zip(actual_closes, expected_closes):
            assert abs(a - b) < 1e-6

    def test_event_count_increments(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="T")
            await stream.collect_all()
            return stream.event_count
        count = asyncio.run(_run())
        assert count == len(small_ohlcv)

    def test_disconnect_sets_closed(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="T")
            await stream.connect()
            await stream.disconnect()
            assert stream.status == StreamStatus.CLOSED
        asyncio.run(_run())

    def test_context_manager(self, small_ohlcv):
        async def _run():
            async with MockMarketStream(small_ohlcv, ticker="T") as stream:
                assert stream.is_connected
            assert stream.status == StreamStatus.CLOSED
        asyncio.run(_run())

    def test_subscribe_sets_tickers(self, small_ohlcv):
        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="T")
            await stream.connect()
            await stream.subscribe(["AAPL", "MSFT"])
            assert "AAPL" in stream._subscribed_tickers
        asyncio.run(_run())

    def test_stream_integrates_with_async_queue(self, small_ohlcv):
        """MockMarketStream → AsyncEventQueue → dispatch chain."""
        from src.backtester.event_queue import AsyncEventQueue
        from src.backtester.events import EventType

        async def _run():
            stream = MockMarketStream(small_ohlcv, ticker="T")
            queue  = AsyncEventQueue()
            received = []
            queue.register(EventType.MARKET, lambda e: received.append(e))

            await stream.connect()
            async for event in stream.stream():
                await queue.put(event)

            await queue.dispatch_all()
            return received

        received = asyncio.run(_run())
        assert len(received) == len(small_ohlcv)
        assert all(isinstance(e, MarketEvent) for e in received)


class TestStreamConfig:
    def test_default_values(self):
        cfg = StreamConfig(uri="wss://example.com")
        assert cfg.reconnect_attempts == 3
        assert cfg.reconnect_delay_s  == 2.0
        assert cfg.max_queue_size     == 10_000
        assert cfg.tickers            == []

    def test_custom_values(self):
        cfg = StreamConfig(
            uri="wss://example.com",
            tickers=["AAPL"],
            reconnect_attempts=5,
            max_queue_size=500,
        )
        assert cfg.tickers            == ["AAPL"]
        assert cfg.reconnect_attempts == 5
        assert cfg.max_queue_size     == 500
