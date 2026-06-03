"""
Performance analytics module.

All metrics follow standard institutional definitions.
Formulas are commented inline for clarity and resume-readiness.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_all_metrics(
    equity_curve: pd.DataFrame,
    returns_col: str = "Daily_Return",
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute the full suite of performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve : pd.DataFrame
        Must contain 'Portfolio_Value' and a daily returns column.
    returns_col : str
        Name of the daily returns column.
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.04 = 4 %). Default 0.

    Returns
    -------
    dict
        Complete metrics dictionary.
    """
    if equity_curve.empty:
        return {}

    pv = equity_curve["Portfolio_Value"].dropna()
    rets = (
        equity_curve[returns_col].dropna()
        if returns_col in equity_curve.columns
        else pv.pct_change().dropna()
    )

    metrics = {}
    metrics.update(_return_metrics(pv, rets))
    metrics.update(_risk_metrics(rets))
    metrics.update(_ratio_metrics(rets, risk_free_rate))
    metrics.update(_trade_metrics(rets))
    metrics.update(_advanced_metrics(rets, risk_free_rate))

    return metrics


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def _return_metrics(pv: pd.Series, rets: pd.Series) -> dict:
    """Cumulative and annualised return metrics."""
    start_val = pv.iloc[0]
    end_val = pv.iloc[-1]
    n_days = len(rets)

    # Cumulative return: (V_final / V_initial) - 1
    cumulative_return = end_val / start_val - 1

    # CAGR: (V_final / V_initial)^(252 / n_days) - 1
    annualised_return = (
        (end_val / start_val) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
        if n_days > 0 else 0.0
    )

    return {
        "Cumulative_Return": round(cumulative_return, 6),
        "Annualised_Return": round(annualised_return, 6),
        "Total_Days": n_days,
        "Start_Value": round(float(start_val), 2),
        "End_Value": round(float(end_val), 2),
    }


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _risk_metrics(rets: pd.Series) -> dict:
    """Volatility, drawdown, Ulcer Index."""
    # Annualised volatility: σ_daily * sqrt(252)
    annualised_vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Maximum drawdown
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Ulcer Index:  sqrt(mean(drawdown^2))
    # Penalises prolonged drawdowns more than VaR does.
    ulcer_index = float(np.sqrt((drawdowns ** 2).mean()))

    # Average drawdown duration (number of bars in drawdown)
    in_dd = (drawdowns < 0)
    dd_lengths = []
    current = 0
    for flag in in_dd:
        if flag:
            current += 1
        elif current > 0:
            dd_lengths.append(current)
            current = 0
    avg_dd_duration = float(np.mean(dd_lengths)) if dd_lengths else 0.0

    return {
        "Annualised_Volatility": round(float(annualised_vol), 6),
        "Max_Drawdown": round(float(max_drawdown), 6),
        "Ulcer_Index": round(ulcer_index, 6),
        "Avg_Drawdown_Duration_Days": round(avg_dd_duration, 1),
    }


# ---------------------------------------------------------------------------
# Ratio metrics
# ---------------------------------------------------------------------------

def _ratio_metrics(rets: pd.Series, risk_free_rate: float) -> dict:
    """Sharpe, Sortino, Calmar, Serenity ratios."""
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = rets - daily_rf

    # Sharpe ratio: E[r - rf] / σ(r - rf) * sqrt(252)
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if excess.std() > 0 else 0.0
    )

    # Sortino ratio: E[r - rf] / σ_downside * sqrt(252)
    downside = rets[rets < daily_rf] - daily_rf
    downside_std = downside.std() if len(downside) > 1 else 0.0
    sortino = (
        excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0 else 0.0
    )

    # Calmar ratio: CAGR / |Max Drawdown|
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())
    n_days = len(rets)
    start = cum.iloc[0] if len(cum) > 0 else 1.0
    end = cum.iloc[-1] if len(cum) > 0 else 1.0
    cagr = (end / start) ** (TRADING_DAYS_PER_YEAR / n_days) - 1 if n_days > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Serenity ratio: CAGR / Ulcer Index  (reward per unit of pain)
    ulcer = float(np.sqrt((drawdowns ** 2).mean()))
    serenity = cagr / ulcer if ulcer > 0 else 0.0

    return {
        "Sharpe_Ratio": round(float(sharpe), 4),
        "Sortino_Ratio": round(float(sortino), 4),
        "Calmar_Ratio": round(float(calmar), 4),
        "Serenity_Ratio": round(float(serenity), 4),
    }


# ---------------------------------------------------------------------------
# Trade metrics
# ---------------------------------------------------------------------------

def _trade_metrics(rets: pd.Series) -> dict:
    """Win rate, average win/loss, profit factor."""
    active = rets[rets != 0]
    if active.empty:
        return {
            "Win_Rate": 0.0,
            "Avg_Win": 0.0,
            "Avg_Loss": 0.0,
            "Profit_Factor": 0.0,
            "Best_Day": 0.0,
            "Worst_Day": 0.0,
        }

    wins = active[active > 0]
    losses = active[active < 0]

    win_rate = len(wins) / len(active)
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "Win_Rate": round(float(win_rate), 4),
        "Avg_Win": round(float(avg_win), 6),
        "Avg_Loss": round(float(avg_loss), 6),
        "Profit_Factor": round(float(profit_factor), 4),
        "Best_Day": round(float(active.max()), 6),
        "Worst_Day": round(float(active.min()), 6),
    }


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------

def _advanced_metrics(rets: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Omega ratio, Tail ratio, Gain-to-Pain, Payoff ratio.
    """
    clean = rets.dropna()
    if len(clean) < 10:
        return {}

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    threshold = daily_rf

    # ---- Omega Ratio ----
    # Ω(r, L) = ∫_L^∞ [1 - F(r)] dr  /  ∫_{-∞}^L F(r) dr
    # Discretised: sum of returns above threshold / |sum of returns below threshold|
    gains = clean[clean > threshold] - threshold
    losses = threshold - clean[clean <= threshold]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else float("inf")

    # ---- Tail Ratio ----
    # P95 / |P5|  —  ratio of right tail to left tail.
    # > 1 means the positive tail is fatter than the negative tail.
    p95 = float(np.percentile(clean, 95))
    p5 = float(np.percentile(clean, 5))
    tail_ratio = abs(p95 / p5) if p5 != 0 else 0.0

    # ---- Gain-to-Pain Ratio ----
    # Sum of all returns / |sum of all negative returns|
    # Similar to profit factor but on continuous return sums.
    total_return = clean.sum()
    total_pain = abs(clean[clean < 0].sum())
    gain_to_pain = total_return / total_pain if total_pain > 0 else 0.0

    # ---- Payoff Ratio ----
    # avg_win / |avg_loss|
    wins = clean[clean > 0]
    losses_neg = clean[clean < 0]
    payoff = (
        abs(wins.mean() / losses_neg.mean())
        if len(wins) > 0 and len(losses_neg) > 0 else 0.0
    )

    # ---- Common Sense Ratio ----
    # Profit Factor × Tail Ratio  (combines distributional and frequency info)
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses_neg.sum()) if len(losses_neg) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    common_sense_ratio = pf * tail_ratio

    return {
        "Omega_Ratio": round(float(omega), 4),
        "Tail_Ratio": round(float(tail_ratio), 4),
        "Gain_to_Pain": round(float(gain_to_pain), 4),
        "Payoff_Ratio": round(float(payoff), 4),
        "Common_Sense_Ratio": round(float(common_sense_ratio), 4),
    }


# ---------------------------------------------------------------------------
# Monthly / Annual returns
# ---------------------------------------------------------------------------

def monthly_returns_table(rets: pd.Series) -> pd.DataFrame:
    """
    Build a Year × Month pivot table of compounded monthly returns.

    Parameters
    ----------
    rets : pd.Series
        Daily return series with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = years, columns = Jan…Dec + 'Annual'.
        Values are fractional returns (multiply by 100 for percentages).
    """
    if rets.empty:
        return pd.DataFrame()

    # Resample to month-end, compound daily returns within each month
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    df = pd.DataFrame({
        "Return": monthly.values,
        "Year": monthly.index.year,
        "Month": monthly.index.month,
    })

    pivot = df.pivot(index="Year", columns="Month", values="Return")
    # Rename integer month columns to abbreviated names
    existing_months = {m: MONTHS[m - 1] for m in pivot.columns if m <= 12}
    pivot.rename(columns=existing_months, inplace=True)

    # Ensure all 12 months present (fill missing with NaN)
    for month in MONTHS:
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[MONTHS]

    # Annual column: compound all monthly returns for the year
    pivot["Annual"] = pivot.apply(
        lambda row: (1 + row.dropna()).prod() - 1, axis=1
    )

    return pivot


def annual_returns(rets: pd.Series) -> pd.Series:
    """
    Compound annual returns by calendar year.

    Parameters
    ----------
    rets : pd.Series
        Daily return series with DatetimeIndex.

    Returns
    -------
    pd.Series
        Annual return indexed by year (integer).
    """
    if rets.empty:
        return pd.Series(dtype=float)
    annual = rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual.index = annual.index.year
    return annual


# ---------------------------------------------------------------------------
# Standalone helpers (used by other modules)
# ---------------------------------------------------------------------------

def cumulative_returns(rets: pd.Series) -> pd.Series:
    """Compute cumulative return series: (1+r1)(1+r2)... - 1"""
    return (1 + rets).cumprod() - 1


def drawdown_series(rets: pd.Series) -> pd.Series:
    """Compute the drawdown series relative to prior peak."""
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def rolling_sharpe(
    rets: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Rolling annualised Sharpe ratio.

    Parameters
    ----------
    rets : pd.Series
    window : int
        Rolling window in trading days.
    risk_free_rate : float
        Annualised risk-free rate.

    Returns
    -------
    pd.Series
    """
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = rets - daily_rf
    roll_mean = excess.rolling(window=window).mean()
    roll_std = excess.rolling(window=window).std()
    return (roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)).where(roll_std > 0)


def best_worst_periods(rets: pd.Series, top_n: int = 5) -> dict:
    """
    Return the top_n best and worst single-day returns.

    Returns
    -------
    dict with keys 'best' and 'worst', each a pd.DataFrame.
    """
    clean = rets.dropna().sort_values(ascending=False)
    best = clean.head(top_n).reset_index()
    best.columns = ["Date", "Return"]
    worst = clean.tail(top_n).sort_values().reset_index()
    worst.columns = ["Date", "Return"]
    return {"best": best, "worst": worst}
