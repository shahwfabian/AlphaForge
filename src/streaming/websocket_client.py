"""
WebSocket-compatible market data stream client.

Production-ready architecture stub for connecting to real-time market
data providers via WebSocket (e.g. Polygon.io, Alpaca, Binance, Kraken).

This module provides:
1. ``WebSocketMarketStream`` — a concrete implementation of
   ``MarketDataStream`` using an async WebSocket connection.
2. A pluggable message parser interface so different providers
   can be supported by implementing ``_parse_message()``.
3. Automatic reconnection with exponential backoff.
4. Heartbeat / ping management to keep connections alive.

Usage (requires 'websockets' package)
--------------------------------------
    pip install websockets

    config = StreamConfig(
        uri="wss://stream.provider.com/v1/ws",
        tickers=["AAPL", "MSFT"],
    )
    stream = WebSocketMarketStream(config, api_key="YOUR_KEY")

    async with stream as s:
        async for event in s.stream():
            await queue.put(event)

To implement a new provider, subclass ``WebSocketMarketStream`` and
override ``_parse_message()`` and optionally ``_build_subscribe_message()``.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from .interfaces import MarketDataStream, StreamConfig, StreamStatus
from ..backtester.events import MarketEvent

logger = logging.getLogger(__name__)


class WebSocketMarketStream(MarketDataStream):
    """
    Async WebSocket market data stream with automatic reconnection.

    Parameters
    ----------
    config : StreamConfig
        Connection configuration (URI, tickers, reconnect settings).
    api_key : str or None
        Optional API key passed in the subscribe message.
    """

    def __init__(
        self,
        config: StreamConfig,
        api_key: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self._api_key    = api_key
        self._ws         = None          # websockets.WebSocketClientProtocol
        self._reconnects = 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the WebSocket connection.

        Raises
        ------
        ImportError
            If the ``websockets`` package is not installed.
        ConnectionError
            If the connection cannot be established after retries.
        """
        try:
            import websockets
        except ImportError as e:
            raise ImportError(
                "The 'websockets' package is required for WebSocketMarketStream. "
                "Install it with: pip install websockets"
            ) from e

        self._set_status(StreamStatus.CONNECTING)
        for attempt in range(self.config.reconnect_attempts):
            try:
                self._ws = await websockets.connect(
                    self.config.uri,
                    ping_interval=self.config.heartbeat_interval_s or None,
                    open_timeout=10,
                )
                self._connected_at = datetime.now(timezone.utc)
                self._set_status(StreamStatus.CONNECTED)
                logger.info(
                    "WebSocket connected: %s (attempt %d/%d)",
                    self.config.uri, attempt + 1, self.config.reconnect_attempts,
                )
                return
            except Exception as e:
                logger.warning(
                    "WebSocket connect attempt %d/%d failed: %s",
                    attempt + 1, self.config.reconnect_attempts, e,
                )
                if attempt < self.config.reconnect_attempts - 1:
                    backoff = self.config.reconnect_delay_s * (2 ** attempt)
                    await asyncio.sleep(backoff)

        self._set_status(StreamStatus.ERROR)
        raise ConnectionError(
            f"Could not connect to {self.config.uri} after "
            f"{self.config.reconnect_attempts} attempts."
        )

    async def subscribe(self, tickers: list[str]) -> None:
        """Send subscription message to the WebSocket provider."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")

        msg = self._build_subscribe_message(tickers)
        await self._ws.send(json.dumps(msg))
        self._subscribed_tickers = tickers
        self._set_status(StreamStatus.SUBSCRIBED)
        logger.info("Subscribed to %s.", tickers)

    async def stream(self) -> AsyncIterator[MarketEvent]:
        """
        Async generator that yields MarketEvent objects from the WebSocket.

        Handles:
        - Automatic reconnection on disconnect.
        - Backoff between reconnection attempts.
        - Graceful stop when status transitions to CLOSED.
        """
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")

        self._set_status(StreamStatus.STREAMING)

        while self.is_connected:
            try:
                raw = await self._ws.recv()
                event = self._parse_message(raw)
                if event is not None:
                    self._increment_event_count()
                    yield event

            except Exception as e:
                if self._status == StreamStatus.CLOSED:
                    break
                logger.warning("WebSocket stream error: %s. Reconnecting…", e)
                self._set_status(StreamStatus.ERROR)
                self._reconnects += 1

                if self._reconnects > self.config.reconnect_attempts:
                    logger.error("Max reconnection attempts exceeded. Stopping.")
                    self._set_status(StreamStatus.CLOSED)
                    break

                backoff = self.config.reconnect_delay_s * (2 ** self._reconnects)
                await asyncio.sleep(min(backoff, 60.0))
                try:
                    await self.connect()
                    await self.subscribe(self._subscribed_tickers)
                except ConnectionError:
                    break

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._set_status(StreamStatus.CLOSED)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info(
            "WebSocket disconnected. Events: %d | Reconnects: %d",
            self._event_count, self._reconnects,
        )

    # ------------------------------------------------------------------
    # Provider-specific message handling (override in subclass)
    # ------------------------------------------------------------------

    def _build_subscribe_message(self, tickers: list[str]) -> dict:
        """
        Build the subscription JSON message for this provider.

        Override in a provider-specific subclass.
        Default format matches a generic subscription pattern.

        Example (Polygon.io):
            {"action": "subscribe", "params": "T.AAPL,T.MSFT"}
        """
        return {
            "action": "subscribe",
            "params": ",".join([f"T.{t}" for t in tickers]),
            **({"apiKey": self._api_key} if self._api_key else {}),
        }

    def _parse_message(self, raw: str) -> MarketEvent | None:
        """
        Parse a raw WebSocket message string into a MarketEvent.

        Override in a provider-specific subclass to match the provider's
        actual message format.

        Default implementation expects a JSON object with fields:
            ev, sym, o (open), h (high), l (low), c (close), v (volume), t (timestamp ms)

        Returns None for non-trade messages (e.g. status, heartbeat).
        """
        try:
            msg = json.loads(raw)

            # Handle list-wrapped messages (e.g. Polygon.io wraps in array)
            if isinstance(msg, list):
                msg = msg[0] if msg else {}

            ev = msg.get("ev", "")
            if ev not in ("T", "A", "AM", "trade", "bar", "quote"):
                return None  # Not a trade/bar message

            ticker = msg.get("sym", msg.get("S", "UNKNOWN"))
            ts_ms  = msg.get("t", msg.get("timestamp", 0))
            ts = (
                datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if ts_ms else datetime.now(timezone.utc)
            )

            return MarketEvent(
                timestamp=ts,
                ticker=ticker,
                open_= float(msg.get("o", msg.get("open",  0.0))),
                high=  float(msg.get("h", msg.get("high",  0.0))),
                low=   float(msg.get("l", msg.get("low",   0.0))),
                close= float(msg.get("c", msg.get("close", msg.get("p", 0.0)))),
                volume=float(msg.get("v", msg.get("volume", 0.0))),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Could not parse message: %s — %s", raw[:100], e)
            return None
