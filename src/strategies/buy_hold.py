"""
Buy-and-Hold Strategy.

The simplest possible strategy: buy at the first available date and
hold through to the last date.  Used as a baseline benchmark.
"""

import pandas as pd

from .base_strategy import BaseStrategy


class BuyAndHold(BaseStrategy):
    """
    Passive buy-and-hold benchmark strategy.

    Always fully invested (+1) from the first trading day onward.
    No parameters required.
    """

    def __init__(self) -> None:
        super().__init__(name="Buy and Hold", parameters={})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        # No parameters to validate
        pass

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Always-long signal: Signal = +1 for every row.

        Notes
        -----
        Even though lookahead bias is not a concern for a trivial buy-and-hold,
        we still apply the 1-day shift for consistency with the rest of the
        strategy framework — the first day is effectively flat.
        """
        out = self._base_output(df)
        out["Signal"] = 1  # Always long
        out = self._finalise(out)
        return out
