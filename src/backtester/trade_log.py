"""
Trade logging module.

Records every order/fill event and exposes a clean DataFrame for
dashboard display and CSV export.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from .events import FillEvent, OrderDirection


@dataclass
class TradeRecord:
    """Single row in the trade log."""
    date: datetime
    ticker: str
    signal: int          # +1 long / 0 flat / -1 short
    direction: str       # "BUY" / "SELL" / "HOLD"
    quantity: float
    fill_price: float
    commission: float
    trade_value: float   # qty * fill_price
    portfolio_value: float


class TradeLog:
    """
    Accumulates TradeRecords throughout a backtest run and provides
    summary statistics and export helpers.
    """

    def __init__(self) -> None:
        self._records: list[TradeRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        fill: FillEvent,
        signal: int,
        portfolio_value: float,
    ) -> None:
        """Append a fill event to the log."""
        record = TradeRecord(
            date=fill.timestamp,
            ticker=fill.ticker,
            signal=signal,
            direction=fill.direction.value,
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            commission=fill.commission,
            trade_value=fill.fill_value,
            portfolio_value=portfolio_value,
        )
        self._records.append(record)

    def record_hold(
        self,
        date: datetime,
        ticker: str,
        portfolio_value: float,
    ) -> None:
        """Record a day where the position is unchanged (HOLD)."""
        self._records.append(
            TradeRecord(
                date=date,
                ticker=ticker,
                signal=1,
                direction="HOLD",
                quantity=0.0,
                fill_price=0.0,
                commission=0.0,
                trade_value=0.0,
                portfolio_value=portfolio_value,
            )
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade records to a clean DataFrame."""
        if not self._records:
            return pd.DataFrame()

        rows = [
            {
                "Date": r.date,
                "Ticker": r.ticker,
                "Signal": r.signal,
                "Direction": r.direction,
                "Quantity": round(r.quantity, 4),
                "Fill_Price": round(r.fill_price, 4),
                "Commission": round(r.commission, 4),
                "Trade_Value": round(r.trade_value, 2),
                "Portfolio_Value": round(r.portfolio_value, 2),
            }
            for r in self._records
        ]
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def count_trades(self) -> int:
        """Return the number of actual BUY/SELL events (not HOLD)."""
        return sum(1 for r in self._records if r.direction in ("BUY", "SELL"))

    def __len__(self) -> int:
        return len(self._records)
