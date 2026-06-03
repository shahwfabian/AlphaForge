"""
AlphaForge — Correlation and Diversification Analytics
=======================================================
Analyses pairwise correlation, portfolio-level diversification,
and allocation concentration for multi-asset portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation matrix from daily returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Columns = tickers, values = daily returns.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.  Empty DataFrame if < 2 columns.
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        return pd.DataFrame()
    return returns_df.dropna().corr().round(4)


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    Mean of all unique off-diagonal correlations.

    Returns NaN if fewer than 2 assets are present.
    """
    n = len(corr_matrix)
    if n < 2:
        return float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = corr_matrix.values[mask]
    return float(np.nanmean(vals))


def most_correlated_pair(corr_matrix: pd.DataFrame) -> tuple[str, str, float]:
    """
    Return (ticker_a, ticker_b, correlation) for the most correlated pair.
    """
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return ("", "", float("nan"))
    mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
    masked = corr_matrix.where(pd.DataFrame(mask, index=corr_matrix.index,
                                            columns=corr_matrix.columns))
    stacked = masked.stack()
    if stacked.empty:
        return ("", "", float("nan"))
    idx = stacked.idxmax()
    return (str(idx[0]), str(idx[1]), float(stacked[idx]))


def least_correlated_pair(corr_matrix: pd.DataFrame) -> tuple[str, str, float]:
    """
    Return (ticker_a, ticker_b, correlation) for the least correlated pair.
    """
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return ("", "", float("nan"))
    mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
    masked = corr_matrix.where(pd.DataFrame(mask, index=corr_matrix.index,
                                            columns=corr_matrix.columns))
    stacked = masked.stack()
    if stacked.empty:
        return ("", "", float("nan"))
    idx = stacked.idxmin()
    return (str(idx[0]), str(idx[1]), float(stacked[idx]))


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """
    Annualised portfolio volatility via the covariance matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame
    weights : dict[str, float]

    Returns
    -------
    float
        Annualised volatility (NaN if data unavailable).
    """
    from src.portfolio.allocation import normalize_weights

    available = {k: v for k, v in weights.items() if k in returns_df.columns}
    if not available:
        return float("nan")

    norm  = normalize_weights(available)
    cols  = list(norm.keys())
    w_arr = np.array([norm[c] for c in cols])
    cov   = returns_df[cols].dropna().cov().values * 252  # annualise

    return float(np.sqrt(w_arr @ cov @ w_arr))


def diversification_ratio(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """
    Diversification Ratio = weighted-avg individual vols / portfolio vol.

    DR > 1 indicates diversification benefit.
    DR = 1 means assets are perfectly correlated (no benefit).
    """
    from src.portfolio.allocation import normalize_weights

    available = {k: v for k, v in weights.items() if k in returns_df.columns}
    if not available:
        return float("nan")

    norm  = normalize_weights(available)
    cols  = list(norm.keys())
    indiv = returns_df[cols].std() * np.sqrt(252)
    wt_avg_vol = sum(norm[c] * float(indiv[c]) for c in cols)
    port_vol   = portfolio_volatility(returns_df, weights)

    if port_vol == 0 or np.isnan(port_vol):
        return float("nan")

    return float(wt_avg_vol / port_vol)


def concentration_score(weights: dict[str, float]) -> float:
    """
    Herfindahl–Hirschman Index (HHI) as a portfolio concentration measure.

    Range: 1/N (fully equal-weighted) to 1.0 (single-asset portfolio).
    High HHI → concentrated, high idiosyncratic risk.
    """
    from src.portfolio.allocation import normalize_weights

    if not weights:
        return float("nan")
    norm = normalize_weights(weights)
    return float(sum(w ** 2 for w in norm.values()))


# ── Full report ───────────────────────────────────────────────────────────────

def diversification_report(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
) -> dict:
    """
    Combine all diversification metrics into a single result dict.

    Returns
    -------
    dict with keys:
        corr_matrix, avg_correlation, portfolio_vol_ann,
        diversification_ratio, concentration_score (HHI),
        most_correlated, least_correlated.
    """
    corr = correlation_matrix(returns_df)
    return {
        "corr_matrix":           corr,
        "avg_correlation":       average_pairwise_correlation(corr),
        "portfolio_vol_ann":     portfolio_volatility(returns_df, weights),
        "diversification_ratio": diversification_ratio(returns_df, weights),
        "concentration_score":   concentration_score(weights),
        "most_correlated":       most_correlated_pair(corr),
        "least_correlated":      least_correlated_pair(corr),
    }
