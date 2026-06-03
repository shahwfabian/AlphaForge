"""
Time-Series Momentum Strategy.

Logic
-----
- Compute trailing total return over a `lookback` period.
- Signal = +1 (long) when trailing return is positive.
- Signal =  0 (flat) when trailing return is negative.
- Optional: skip the most recent `skip_days` before computing return
  to avoid short-term mean-reversion (standard in academic literature).
- Execution shifted by 1 bar to prevent lookahead bias.
"""

import pandas as pd

from .base_strategy import BaseStrategy


class Momentum(BaseStrategy):
    """
    Time-series momentum (trend-following) strategy.

    Parameters
    ----------
    lookback : int
        Number of trading days used to measure trailing return (default 90).
    skip_days : int
        Most-recent days to exclude from return computation.
        Avoids short-term reversal contamination.  Default 1.
    """

    def __init__(self, lookback: int = 90, skip_days: int = 1) -> None:
        super().__init__(
            name="Momentum",
            parameters={"lookback": lookback, "skip_days": skip_days},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        lookback = self.parameters["lookback"]
        skip = self.parameters["skip_days"]

        if not isinstance(lookback, int) or lookback < 10:
            raise ValueError(f"lookback must be an integer >= 10, got {lookback}.")
        if not isinstance(skip, int) or skip < 0:
            raise ValueError(f"skip_days must be a non-negative integer, got {skip}.")
        if skip >= lookback:
            raise ValueError(
                f"skip_days ({skip}) must be less than lookback ({lookback})."
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-series momentum signals.

        Trailing return from t-lookback to t-skip_days.
        If that return is positive the asset has upward momentum → long.

        Notes
        -----
        Academic reference: Moskowitz, Ooi & Pedersen (2012).
        "Time Series Momentum." Journal of Financial Economics.
        """
        lookback = self.parameters["lookback"]
        skip = self.parameters["skip_days"]

        out = self._base_output(df)

        # Price from `lookback` days ago (the start of the measurement window)
        price_past = df["Close"].shift(lookback)

        # Price from `skip_days` ago (end of measurement window, avoids reversal)
        price_recent = df["Close"].shift(skip)

        # Trailing return = (P_recent - P_past) / P_past
        trailing_return = (price_recent - price_past) / price_past

        out["Trailing_Return"] = trailing_return

        # +1 if trailing return is positive (upward momentum), else 0
        out["Signal"] = (trailing_return > 0).astype(int).where(
            trailing_return.notna(), other=0
        )

        out = self._finalise(out)
        return out
