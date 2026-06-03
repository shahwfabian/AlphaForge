"""
Bollinger Bands Mean-Reversion Strategy.

Logic
-----
- Compute rolling mean (middle band) and rolling std of Close prices.
- Upper Band = SMA + n_std × σ
- Lower Band = SMA − n_std × σ
- Signal = +1 (long) when price closes *below* the lower band.
  → Price is statistically cheap vs its recent distribution.
- Position is exited when price crosses back above the middle band.
- Execution shifted by 1 bar to prevent lookahead bias.

Academic basis
--------------
Bollinger, J. (2002). "Bollinger on Bollinger Bands."
The bands contain ~95% of price action when n_std=2 (for normally distributed returns).
Touches of the lower band signal potential mean-reversion opportunities.
"""

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy


class BollingerBands(BaseStrategy):
    """
    Bollinger Bands mean-reversion strategy.

    Parameters
    ----------
    window : int
        Rolling window for SMA and standard deviation (default 20).
    n_std : float
        Number of standard deviations for band width (default 2.0).
    exit_at_middle : bool
        If True, exit when price crosses the middle band.
        If False, exit when price crosses the upper band.  Default True.
    """

    def __init__(
        self,
        window: int = 20,
        n_std: float = 2.0,
        exit_at_middle: bool = True,
    ) -> None:
        super().__init__(
            name="Bollinger Bands",
            parameters={
                "window": window,
                "n_std": n_std,
                "exit_at_middle": exit_at_middle,
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        window = self.parameters["window"]
        n_std = self.parameters["n_std"]

        if not isinstance(window, int) or window < 5:
            raise ValueError(f"window must be an integer >= 5, got {window}.")
        if not (0.5 <= n_std <= 4.0):
            raise ValueError(f"n_std must be in [0.5, 4.0], got {n_std}.")

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Band signals.

        Entry  : Close < Lower Band  → price is statistically cheap
        Exit   : Close > Middle Band (or Upper Band if exit_at_middle=False)
        """
        window = self.parameters["window"]
        n_std = self.parameters["n_std"]
        exit_at_middle = self.parameters["exit_at_middle"]

        out = self._base_output(df)

        sma = df["Close"].rolling(window=window, min_periods=window).mean()
        std = df["Close"].rolling(window=window, min_periods=window).std()

        out["BB_Middle"] = sma
        out["BB_Upper"] = sma + n_std * std
        out["BB_Lower"] = sma - n_std * std
        out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / sma  # normalized width
        out["BB_Pct"] = (df["Close"] - out["BB_Lower"]) / (out["BB_Upper"] - out["BB_Lower"])

        exit_band = out["BB_Middle"] if exit_at_middle else out["BB_Upper"]

        signals = []
        in_position = False

        for i, (close, lower, exit_b) in enumerate(
            zip(df["Close"], out["BB_Lower"], exit_band)
        ):
            if pd.isna(lower):
                signals.append(0)
                continue

            if not in_position and close < lower:
                # Price pierced lower band → mean-reversion entry
                in_position = True
                signals.append(1)
            elif in_position and close > exit_b:
                # Price recovered to middle (or upper) → exit
                in_position = False
                signals.append(0)
            elif in_position:
                signals.append(1)
            else:
                signals.append(0)

        out["Signal"] = signals
        out = self._finalise(out)
        return out
