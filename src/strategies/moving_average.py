"""
Moving Average Crossover Strategy.

Logic
-----
- Compute a short-window SMA and a long-window SMA.
- Signal = +1 (long) when short SMA > long SMA  (golden cross)
- Signal =  0 (flat) when short SMA < long SMA  (death cross)
- Execution is shifted by 1 bar to prevent lookahead bias.
"""

import pandas as pd

from .base_strategy import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """
    Dual Simple Moving Average (SMA) crossover strategy.

    Parameters
    ----------
    short_window : int
        Number of days for the fast / short moving average (default 20).
    long_window : int
        Number of days for the slow / long moving average (default 50).
    """

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        super().__init__(
            name="Moving Average Crossover",
            parameters={"short_window": short_window, "long_window": long_window},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        short = self.parameters["short_window"]
        long_ = self.parameters["long_window"]

        if not isinstance(short, int) or short < 2:
            raise ValueError(f"short_window must be an integer >= 2, got {short}.")
        if not isinstance(long_, int) or long_ < 2:
            raise ValueError(f"long_window must be an integer >= 2, got {long_}.")
        if short >= long_:
            raise ValueError(
                f"short_window ({short}) must be strictly less than "
                f"long_window ({long_})."
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SMA crossover signals.

        A buy signal (+1) is issued when the short SMA crosses above the long SMA.
        The position is held until the short SMA crosses back below the long SMA.

        Notes
        -----
        The first `long_window` rows will have NaN SMAs and are assigned Signal = 0.
        """
        short_w = self.parameters["short_window"]
        long_w = self.parameters["long_window"]

        out = self._base_output(df)

        # Compute SMAs
        out["SMA_Short"] = df["Close"].rolling(window=short_w, min_periods=short_w).mean()
        out["SMA_Long"] = df["Close"].rolling(window=long_w, min_periods=long_w).mean()

        # Raw signal: +1 when short > long, else 0
        # Using .fillna(0) ensures the warm-up period stays flat
        out["Signal"] = (
            (out["SMA_Short"] > out["SMA_Long"]).astype(int).where(
                out["SMA_Short"].notna() & out["SMA_Long"].notna(), other=0
            )
        )

        # Shift by 1 to avoid lookahead bias; compute strategy returns
        out = self._finalise(out)
        return out
