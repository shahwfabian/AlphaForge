"""
Abstract base class for all AlphaForge trading strategies.

Every concrete strategy must inherit from BaseStrategy and implement:
  - generate_signals(df) -> pd.DataFrame
  - validate_parameters()
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base for systematic trading strategies.

    Attributes
    ----------
    name : str
        Human-readable strategy name.
    parameters : dict
        Strategy-specific hyper-parameters.
    """

    def __init__(self, name: str, parameters: dict[str, Any]) -> None:
        self.name = name
        self.parameters = parameters
        self.validate_parameters()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_parameters(self) -> None:
        """
        Raise ValueError if any parameter is invalid.
        Called automatically at construction time.
        """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry/exit signals from a preprocessed OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least a 'Close' column indexed by date.
            May also contain 'Return', 'Log_Return', etc.

        Returns
        -------
        pd.DataFrame
            Index : same DatetimeIndex as df.
            Columns (required):
              - ``Signal``    : int  (+1 = long, 0 = flat, -1 = short)
              - ``Position``  : int  current position after applying signal
              - ``Asset_Return``    : float  asset daily return
              - ``Strategy_Return`` : float  strategy daily return (position * asset return)
        """

    # ------------------------------------------------------------------
    # Helpers shared by all strategies
    # ------------------------------------------------------------------

    def _base_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the skeleton output DataFrame aligned to *df*'s index.
        Concrete strategies fill in Signal and Position then call
        _finalise() to compute returns.
        """
        out = pd.DataFrame(index=df.index)
        out["Close"] = df["Close"]
        out["Asset_Return"] = df["Return"] if "Return" in df.columns else df["Close"].pct_change()
        out["Signal"] = 0
        out["Position"] = 0
        out["Strategy_Return"] = 0.0
        return out

    @staticmethod
    def _finalise(out: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Strategy_Return from Position and Asset_Return.

        IMPORTANT — lookahead-bias prevention:
            We shift Position by 1 day so we trade at the *next* open,
            i.e. a signal generated on day t is executed on day t+1.
        """
        out["Position"] = out["Signal"].shift(1).fillna(0).astype(int)
        out["Strategy_Return"] = out["Position"] * out["Asset_Return"]
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.parameters})"
