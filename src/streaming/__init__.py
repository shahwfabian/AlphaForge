"""
AlphaForge Streaming Layer.

Provides abstract interfaces and concrete implementations for ingesting
real-time market data into the backtesting/live-trading event pipeline.

Architecture
------------
                 ┌─────────────────────────────┐
    WebSocket/   │   MarketDataStream (ABC)     │
    HTTP Feed ──►│  connect() / subscribe()     │
                 │  stream() → AsyncIterator    │──► AsyncEventQueue
                 └─────────────────────────────┘
                          ▲
               ┌──────────┴───────────┐
               │                      │
         MockStream              WebSocketStream
    (replay historical)     (live WebSocket feed)

Usage
-----
from src.streaming import MockMarketStream
from src.backtester.event_queue import AsyncEventQueue

stream = MockMarketStream(df)
queue  = AsyncEventQueue()

async def run():
    await stream.connect()
    async for event in stream.stream():
        await queue.put(event)
        await queue.dispatch_all()
"""

from .interfaces import MarketDataStream, StreamConfig, StreamStatus
from .mock_stream import MockMarketStream
from .websocket_client import WebSocketMarketStream

__all__ = [
    "MarketDataStream",
    "StreamConfig",
    "StreamStatus",
    "MockMarketStream",
    "WebSocketMarketStream",
]
