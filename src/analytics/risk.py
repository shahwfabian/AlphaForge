"""
Risk analytics module.

Computes Value at Risk (VaR), Expected Shortfall (CVaR),
and rolling risk measures.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical (non-parametric) Value at Risk.

    VaR answers: "What is the maximum loss over `horizon_days` days
    that will not be exceeded with `confidence` probability?"

    Method: sort historical returns and read off the (1-confidence) quantile.

    Parameters
    ----------
    returns : pd.Series
        Daily return series (not cumulative).
    confidence : float
        Confidence level (e.g. 0.95 for 95 % VaR).
    horizon_days : int
        Holding period. Scaled via the square-root-of-time rule.

    Returns
    -------
    float
        VaR as a negative number (e.g. -0.025 means -2.5 % potential loss).
    """
    if returns.empty:
        return 0.0
    clean = returns.dropna()
    quantile_loss = clean.quantile(1 - confidence)
    # Scale to multi-day horizon: VaR_T = VaR_1 * sqrt(T)
    return float(quantile_loss * np.sqrt(horizon_days))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed:
        VaR = μ - z * σ
    where z is the z-score corresponding to (1-confidence).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
    horizon_days : int

    Returns
    -------
    float
        VaR (negative number).
    """
    if returns.empty:
        return 0.0
    clean = returns.dropna()
    mu = clean.mean()
    sigma = clean.std()
    z = stats.norm.ppf(1 - confidence)
    daily_var = mu + z * sigma   # z is negative at left tail
    return float(daily_var * np.sqrt(horizon_days))


# ---------------------------------------------------------------------------
# Expected Shortfall (CVaR)
# ---------------------------------------------------------------------------

def expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical Expected Shortfall (ES / CVaR).

    ES is the average loss in the worst (1-confidence) fraction of scenarios.
    It is a coherent risk measure and is preferred over VaR by Basel III / IV.

        ES = E[r | r < VaR]

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level.
    horizon_days : int
        Holding period (square-root-of-time scaling applied).

    Returns
    -------
    float
        Expected Shortfall (negative number).
    """
    if returns.empty:
        return 0.0
    clean = returns.dropna()
    var = historical_var(clean, confidence, horizon_days=1)
    tail_losses = clean[clean <= var]
    if tail_losses.empty:
        return float(var)
    es = tail_losses.mean()
    return float(es * np.sqrt(horizon_days))


# ---------------------------------------------------------------------------
# Rolling risk measures
# ---------------------------------------------------------------------------

def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """
    Rolling annualised volatility.

    Parameters
    ----------
    returns : pd.Series
    window : int
        Rolling window in trading days.
    annualise : bool
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
    """
    roll_vol = returns.rolling(window=window).std()
    if annualise:
        roll_vol = roll_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return roll_vol


def rolling_var(
    returns: pd.Series,
    window: int = 63,
    confidence: float = 0.95,
) -> pd.Series:
    """Rolling historical VaR over a moving window."""
    return returns.rolling(window=window).quantile(1 - confidence)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Rolling Sharpe ratio over a moving window.

    Annualised: (mean(r - rf) / std(r - rf)) * sqrt(252)
    """
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - daily_rf
    roll_mean = excess.rolling(window=window).mean()
    roll_std = excess.rolling(window=window).std()
    return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Full risk summary
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """
    Compute all risk metrics and return as a dictionary.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        VaR / ES confidence level.

    Returns
    -------
    dict
    """
    clean = returns.dropna()
    if clean.empty:
        return {}

    return {
        "VaR_Historical_95": round(historical_var(clean, 0.95), 6),
        "VaR_Historical_99": round(historical_var(clean, 0.99), 6),
        "VaR_Parametric_95": round(parametric_var(clean, 0.95), 6),
        "Expected_Shortfall_95": round(expected_shortfall(clean, 0.95), 6),
        "Expected_Shortfall_99": round(expected_shortfall(clean, 0.99), 6),
        "Skewness": round(float(clean.skew()), 4),
        "Kurtosis": round(float(clean.kurtosis()), 4),  # Excess kurtosis
        "Daily_Volatility": round(float(clean.std()), 6),
    }
