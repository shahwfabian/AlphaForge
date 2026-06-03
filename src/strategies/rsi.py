"""
RSI (Relative Strength Index) Mean-Reversion Strategy.

Logic
-----
- Compute Wilder's RSI over a rolling `period` window.
- Signal = +1 (long) when RSI < `oversold` threshold  (asset is cheap / momentum negative)
- Position held until RSI rises above `exit_level`    (momentum recovering)
- Signal =  0 (flat) when RSI > `overbought` threshold (optional short-sell guard)
- Execution shifted by 1 bar to prevent lookahead bias.

Academic basis
--------------
Wilder, J.W. (1978). "New Concepts in Technical Trading Systems."
RSI = 100 − 100 / (1 + RS),  RS = avg_gain / avg_loss  (Wilder smoothing).
"""

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    """
    RSI-based mean reversion: buy oversold, exit on recovery.

    Parameters
    ----------
    period : int
        RSI look-back period (Wilder default = 14).
    oversold : float
        RSI level below which a long entry is triggered (default 30).
    overbought : float
        RSI level above which any long position is exited (default 70).
    exit_level : float
        RSI level above which the position is closed if below overbought
        (default 50 — exits on mean recovery, not overbought exhaustion).
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
    ) -> None:
        super().__init__(
            name="RSI Mean Reversion",
            parameters={
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "exit_level": exit_level,
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        period = self.parameters["period"]
        oversold = self.parameters["oversold"]
        overbought = self.parameters["overbought"]
        exit_level = self.parameters["exit_level"]

        if not isinstance(period, int) or period < 2:
            raise ValueError(f"period must be an integer >= 2, got {period}.")
        if not (0 < oversold < 50):
            raise ValueError(f"oversold must be in (0, 50), got {oversold}.")
        if not (50 < overbought < 100):
            raise ValueError(f"overbought must be in (50, 100), got {overbought}.")
        if not (oversold < exit_level < overbought):
            raise ValueError(
                f"exit_level ({exit_level}) must be between oversold ({oversold}) "
                f"and overbought ({overbought})."
            )

    # ------------------------------------------------------------------
    # RSI computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        """
        Wilder's RSI using exponential weighted moving averages.

        Wilder's smoothing:  α = 1 / period  (equivalent to EWM com = period-1).
        RS = EWM_avg_gain / EWM_avg_loss
        RSI = 100 − 100 / (1 + RS)
        """
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Where avg_loss = 0 (every day was a gain), RSI = 100
        rsi = rsi.fillna(100.0)
        return rsi

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI and derive entry/exit signals.

        Entry : RSI < oversold  → long (+1)
        Exit  : RSI > exit_level → flat (0)
        Guard : RSI > overbought → force flat (0) even if already entered
        """
        period = self.parameters["period"]
        oversold = self.parameters["oversold"]
        overbought = self.parameters["overbought"]
        exit_level = self.parameters["exit_level"]

        out = self._base_output(df)
        out["RSI"] = self._compute_rsi(df["Close"], period)

        signals = []
        in_position = False

        for rsi_val in out["RSI"]:
            if pd.isna(rsi_val):
                signals.append(0)
                continue

            if not in_position and rsi_val < oversold:
                in_position = True
                signals.append(1)
            elif in_position and (rsi_val > exit_level or rsi_val > overbought):
                in_position = False
                signals.append(0)
            elif in_position:
                signals.append(1)
            else:
                signals.append(0)

        out["Signal"] = signals
        out = self._finalise(out)
        return out
