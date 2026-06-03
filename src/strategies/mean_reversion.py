"""
Mean Reversion (Z-Score) Strategy.

Logic
-----
- Compute rolling mean and rolling standard deviation of price.
- Z-Score = (Price - Rolling Mean) / Rolling Std
- Signal = +1 (buy) when Z-Score < -entry_threshold  (price cheap vs history)
- Signal =  0 (flat) when Z-Score > exit_threshold   (price reverts toward mean)
- Signal stays +1 between entry and exit to hold the position.
- Execution shifted by 1 bar to prevent lookahead bias.
"""

import pandas as pd

from .base_strategy import BaseStrategy


class MeanReversion(BaseStrategy):
    """
    Statistical mean-reversion strategy based on rolling z-score.

    Parameters
    ----------
    window : int
        Rolling look-back window in trading days (default 20).
    entry_threshold : float
        Z-score level below which a long entry is triggered (default 2.0).
        Must be positive; entry fires when z < -entry_threshold.
    exit_threshold : float
        Z-score level above which the position is closed (default 0.5).
        Must be positive; exit fires when z > -exit_threshold (i.e. reversion).
    """

    def __init__(
        self,
        window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            name="Mean Reversion",
            parameters={
                "window": window,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        window = self.parameters["window"]
        entry = self.parameters["entry_threshold"]
        exit_ = self.parameters["exit_threshold"]

        if not isinstance(window, int) or window < 5:
            raise ValueError(f"window must be an integer >= 5, got {window}.")
        if entry <= 0:
            raise ValueError(f"entry_threshold must be positive, got {entry}.")
        if exit_ <= 0:
            raise ValueError(f"exit_threshold must be positive, got {exit_}.")
        if exit_ >= entry:
            raise ValueError(
                f"exit_threshold ({exit_}) should be less than "
                f"entry_threshold ({entry}) for sensible mean-reversion logic."
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean-reversion signals using rolling z-score.

        The position is long (+1) when z-score drops below -entry_threshold
        and is held until z-score recovers above -exit_threshold.
        """
        window = self.parameters["window"]
        entry_thr = self.parameters["entry_threshold"]
        exit_thr = self.parameters["exit_threshold"]

        out = self._base_output(df)

        rolling_mean = df["Close"].rolling(window=window, min_periods=window).mean()
        rolling_std = df["Close"].rolling(window=window, min_periods=window).std()

        # Avoid division-by-zero
        rolling_std = rolling_std.replace(0, float("nan"))

        out["Z_Score"] = (df["Close"] - rolling_mean) / rolling_std

        # Build signal with position memory (hold between entry and exit)
        signals = []
        in_position = False

        for z in out["Z_Score"]:
            if pd.isna(z):
                signals.append(0)
                continue

            if not in_position and z < -entry_thr:
                # Price is significantly below its rolling mean → buy
                in_position = True
                signals.append(1)
            elif in_position and z > -exit_thr:
                # Price has mean-reverted sufficiently → close
                in_position = False
                signals.append(0)
            elif in_position:
                signals.append(1)
            else:
                signals.append(0)

        out["Signal"] = signals
        out = self._finalise(out)
        return out
