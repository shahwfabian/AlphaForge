"""
Mock market data stream for backtesting and testing.

Replays a historical OHLCV DataFrame as a stream of ``MarketEvent`` objects
at configurable speed.  Implements ``MarketDataStream`` so it is a drop-in
replacement for a live WebSocket feed.

Use cases
---------
- Unit testing async event handlers without network I/O.
- Strategy testing in real-time simulation (speed > 1.0 for fast replay).
- CI/CD pipelines that need a deterministic market data source.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import pandas as pd

from .interfaces import MarketDataStream, StreamConfig, StreamStatus
from ..backtester.events import MarketEvent

logger = logging.getLogger(__name__)


class MockMarketStream(MarketDataStream):
    """
    Replay a historical DataFrame as a live-style market event stream.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed OHLCV DataFrame with DatetimeIndex.
    ticker : str
        Symbol to attach to every emitted MarketEvent.
    replay_speed : float
        Simulated events per second.  0.0 = as fast as possible (no sleep).
    loop_forever : bool
        If True, replay the DataFrame in a loop indefinitely.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker: str = "MOCK",
        replay_speed: float = 0.0,
        loop_forever: bool = False,
    ) -> None:
        super().__init__(config=None)
        self._df           = df.copy()
        self._ticker       = ticker
        self._replay_speed = replay_speed
        self._loop_forever = loop_forever
        self._paused       = asyncio.Event()
        self._paused.set()  # not paused by default

    # ------------------------------------------------------------------
    # MarketDataStream interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._set_status(StreamStatus.CONNECTING)
        await asyncio.sleep(0)  # yield to event loop
        self._connected_at = datetime.now(timezone.utc)
        self._set_status(StreamStatus.CONNECTED)
        logger.info("MockMarketStream connected | ticker=%s | rows=%d",
                    self._ticker, len(self._df))

    async def subscribe(self, tickers: list[str]) -> None:
        self._subscribed_tickers = tickers
        self._set_status(StreamStatus.SUBSCRIBED)

    async def stream(self) -> AsyncIterator[MarketEvent]:
        """
        Async generator that replays the DataFrame row by row.

        Yields
        ------
        MarketEvent
            One event per bar.
        """
        self._set_status(StreamStatus.STREAMING)

        while True:
            for timestamp, row in self._df.iterrows():
                # Honour pause/resume
                await self._paused.wait()

                # Compute inter-event delay
                if self._replay_speed > 0:
                    await asyncio.sleep(1.0 / self._replay_speed)
                else:
                    await asyncio.sleep(0)  # still yield control

                event = MarketEvent(
                    timestamp=timestamp,
                    ticker=self._ticker,
                    open_=float(row.get("Open",   row["Close"])),
                    high= float(row.get("High",   row["Close"])),
                    low=  float(row.get("Low",    row["Close"])),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
                self._increment_event_count()
                yield event

            if not self._loop_forever:
                break

        self._set_status(StreamStatus.CLOSED)

    async def disconnect(self) -> None:
        self._set_status(StreamStatus.CLOSED)
        logger.info("MockMarketStream disconnected after %d events.", self._event_count)

    # ------------------------------------------------------------------
    # Pause / resume (useful for testing)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the stream (event emission suspends until resumed)."""
        self._paused.clear()
        logger.debug("MockMarketStream paused.")

    def resume(self) -> None:
        """Resume a paused stream."""
        self._paused.set()
        logger.debug("MockMarketStream resumed.")

    # ------------------------------------------------------------------
    # Convenience: collect all events (blocking helper for tests)
    # ------------------------------------------------------------------

    async def collect_all(self) -> list[MarketEvent]:
        """
        Replay the entire DataFrame and return all events as a list.
        Only valid when ``loop_forever=False``.
        """
        events = []
        await self.connect()
        async for event in self.stream():
            events.append(event)
        return events
