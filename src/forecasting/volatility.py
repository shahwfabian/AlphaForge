"""
AlphaForge — Volatility Forecasting Engine
===========================================
Three volatility estimators in a unified interface:

1. Rolling Historical Volatility
   σ̂_t = std(r_{t-w+1}, …, r_t) × √252
   Simple, interpretable, lagging.

2. EWMA (RiskMetrics, λ = 0.94)
   σ²_t = λ σ²_{t-1} + (1 - λ) r²_t
   Faster response to recent shocks; industry standard for VaR.

3. GARCH(1,1)
   σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
   Maximum-likelihood estimation via the `arch` library.
   Captures volatility clustering (σ persistence + mean reversion).
   Falls back to EWMA if arch is unavailable.

Outputs include current vol estimates, multi-step-ahead GARCH forecasts,
annualised series, and regime-coloured volatility bands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── GARCH availability guard ──────────────────────────────────────────────────
try:
    from arch import arch_model as _arch_model
    _GARCH_AVAILABLE = True
except ImportError:
    _GARCH_AVAILABLE = False
    logger.warning("arch library not available — GARCH falls back to EWMA.")

_DISCLAIMER = (
    "Volatility forecasts are statistical estimates based on historical "
    "return behaviour.  Realised future volatility may differ materially, "
    "especially around structural breaks or news events."
)

EWMA_LAMBDA_DEFAULT = 0.94   # RiskMetrics λ


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class VolForecastResult:
    """Unified volatility forecast container."""

    ticker: str
    method: str                         # "Rolling" | "EWMA" | "GARCH"

    # Historical series (annualised)
    hist_vol: pd.Series                 # σ̂_t for each historical bar

    # Current estimates
    current_vol_ann: float              # latest annualised vol
    current_vol_daily: float

    # Multi-step-ahead forecasts (GARCH or flat projection)
    forecast_horizon: int
    forecast_vol_ann: pd.Series         # (horizon,) forward vol curve
    forecast_ci_lo:   pd.Series         # 95% lower band
    forecast_ci_hi:   pd.Series         # 95% upper band

    # Regime classification
    vol_regime: str                     # "Low" | "Normal" | "Elevated" | "High"
    vol_pct_rank: float                 # percentile of current vol in history

    # GARCH params (NaN if not used)
    garch_omega: float = float("nan")
    garch_alpha: float = float("nan")
    garch_beta:  float = float("nan")
    garch_hl:    float = float("nan")   # half-life of vol shock (days)

    disclaimer: str = field(default=_DISCLAIMER, repr=False)

    def summary(self) -> dict:
        d = {
            "Method":           self.method,
            "Current Vol (ann)":f"{self.current_vol_ann * 100:.2f}%",
            "Vol Regime":       self.vol_regime,
            "Percentile Rank":  f"{self.vol_pct_rank * 100:.0f}th",
        }
        if not np.isnan(self.garch_beta):
            d["GARCH α + β"]   = f"{(self.garch_alpha + self.garch_beta):.4f}"
            d["Vol Half-Life"]  = f"{self.garch_hl:.1f} days"
        return d


# ── Rolling volatility ────────────────────────────────────────────────────────

def rolling_vol(
    prices: pd.Series,
    window: int = 21,
    forecast_horizon: int = 21,
) -> VolForecastResult:
    """Historical rolling window volatility with flat forward projection."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    hist    = log_ret.rolling(window, min_periods=window).std() * np.sqrt(252)
    hist    = hist.dropna()

    current_ann   = float(hist.iloc[-1])
    current_daily = current_ann / np.sqrt(252)
    pct_rank      = float(stats.percentileofscore(hist.values, current_ann) / 100)

    last_date = prices.index[-1]
    fwd_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    flat_vol  = pd.Series(current_ann, index=fwd_dates)

    # Naive CI from historical distribution of rolling vol
    vol_std   = float(hist.std())
    ci_lo     = pd.Series(np.maximum(0, current_ann - 1.96 * vol_std), index=fwd_dates)
    ci_hi     = pd.Series(current_ann + 1.96 * vol_std, index=fwd_dates)

    return VolForecastResult(
        ticker            = str(prices.name or "ASSET"),
        method            = f"Rolling ({window}d)",
        hist_vol          = hist,
        current_vol_ann   = current_ann,
        current_vol_daily = current_daily,
        forecast_horizon  = forecast_horizon,
        forecast_vol_ann  = flat_vol,
        forecast_ci_lo    = ci_lo,
        forecast_ci_hi    = ci_hi,
        vol_regime        = _vol_regime(pct_rank),
        vol_pct_rank      = pct_rank,
    )


# ── EWMA volatility ───────────────────────────────────────────────────────────

def ewma_vol(
    prices: pd.Series,
    lam: float = EWMA_LAMBDA_DEFAULT,
    forecast_horizon: int = 21,
) -> VolForecastResult:
    """
    RiskMetrics EWMA volatility.

    σ²_t = λ σ²_{t-1} + (1-λ) r²_t
    """
    log_ret = np.log(prices / prices.shift(1)).dropna().values
    T       = len(log_ret)
    var     = np.zeros(T)
    var[0]  = log_ret[0] ** 2
    for t in range(1, T):
        var[t] = lam * var[t - 1] + (1 - lam) * log_ret[t] ** 2

    hist_daily = pd.Series(np.sqrt(var), index=prices.index[1:])
    hist_ann   = hist_daily * np.sqrt(252)

    current_daily = float(hist_daily.iloc[-1])
    current_ann   = current_daily * np.sqrt(252)
    pct_rank      = float(stats.percentileofscore(hist_ann.values, current_ann) / 100)

    # EWMA forecast: σ²_{t+h} = lam^h σ²_t + (1-lam^h)/(1-lam) * (1-lam)*σ̄²
    # Simplified: decay toward long-run vol
    long_run_var  = float(log_ret.var())
    last_var      = current_daily ** 2
    last_date     = prices.index[-1]
    fwd_dates     = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    fwd_var = np.array([
        lam ** h * last_var + (1 - lam ** h) * long_run_var
        for h in range(1, forecast_horizon + 1)
    ])
    fwd_vol = pd.Series(np.sqrt(fwd_var) * np.sqrt(252), index=fwd_dates)

    vol_std = float(hist_ann.std())
    ci_lo   = (fwd_vol - 1.96 * vol_std).clip(lower=0)
    ci_hi   = fwd_vol + 1.96 * vol_std

    return VolForecastResult(
        ticker            = str(prices.name or "ASSET"),
        method            = f"EWMA (λ={lam})",
        hist_vol          = hist_ann,
        current_vol_ann   = current_ann,
        current_vol_daily = current_daily,
        forecast_horizon  = forecast_horizon,
        forecast_vol_ann  = fwd_vol,
        forecast_ci_lo    = pd.Series(ci_lo.values, index=fwd_dates),
        forecast_ci_hi    = pd.Series(ci_hi.values, index=fwd_dates),
        vol_regime        = _vol_regime(pct_rank),
        vol_pct_rank      = pct_rank,
    )


# ── GARCH(1,1) volatility ─────────────────────────────────────────────────────

def garch_vol(
    prices: pd.Series,
    forecast_horizon: int = 21,
    p: int = 1,
    q: int = 1,
) -> VolForecastResult:
    """
    GARCH(p,q) volatility estimation using the `arch` library.

    Falls back to EWMA if arch is not available or fitting fails.

    Model:  r_t   = ε_t,   ε_t ~ N(0, σ²_t)
            σ²_t  = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Annualised vol forecast uses:
        E[σ²_{t+h}] = (ω/(1-α-β)) + (α+β)^h (σ²_t - ω/(1-α-β))
    """
    if not _GARCH_AVAILABLE:
        logger.warning("arch unavailable; falling back to EWMA for GARCH call.")
        res = ewma_vol(prices, forecast_horizon=forecast_horizon)
        res.method = "GARCH(1,1) [EWMA fallback]"
        return res

    log_ret = np.log(prices / prices.shift(1)).dropna() * 100  # arch uses pct
    ticker  = str(prices.name or "ASSET")

    try:
        am  = _arch_model(log_ret, vol="Garch", p=p, q=q, dist="Normal")
        res_fit = am.fit(disp="off", show_warning=False)

        omega = float(res_fit.params.get("omega", 0.01))
        alpha = float(res_fit.params.get(f"alpha[1]", 0.05))
        beta  = float(res_fit.params.get(f"beta[1]",  0.90))
        ab    = alpha + beta

        # Conditional variance history (daily, in pct² units)
        cond_vol_pct = res_fit.conditional_volatility            # pct/day
        hist_ann     = (cond_vol_pct / 100) * np.sqrt(252)
        hist_ann.index = log_ret.index

        current_daily = float(cond_vol_pct.iloc[-1] / 100)
        current_ann   = current_daily * np.sqrt(252)
        pct_rank      = float(
            stats.percentileofscore(hist_ann.values, current_ann) / 100
        )

        # Multi-step GARCH variance forecast
        last_sigma2  = current_daily ** 2
        long_run_var = omega / max(1.0 - ab, 1e-6) / 1e4   # back to decimal

        last_date = prices.index[-1]
        fwd_dates = pd.bdate_range(last_date + pd.Timedelta(days=1),
                                   periods=forecast_horizon)

        fwd_var = np.array([
            long_run_var + ab ** h * (last_sigma2 - long_run_var)
            for h in range(1, forecast_horizon + 1)
        ])
        fwd_var = np.maximum(fwd_var, 1e-10)
        fwd_vol = pd.Series(np.sqrt(fwd_var) * np.sqrt(252), index=fwd_dates)

        # Forecast CI via bootstrap residuals
        vol_std = float(hist_ann.std())
        ci_lo   = (fwd_vol - 1.96 * vol_std).clip(lower=0)
        ci_hi   = fwd_vol + 1.96 * vol_std

        # Half-life of shock: (α+β)^h = 0.5  →  h = log(0.5) / log(α+β)
        hl = np.log(0.5) / np.log(max(ab, 1e-9)) if ab > 0 and ab < 1 else float("nan")

        return VolForecastResult(
            ticker            = ticker,
            method            = f"GARCH({p},{q})",
            hist_vol          = hist_ann,
            current_vol_ann   = current_ann,
            current_vol_daily = current_daily,
            forecast_horizon  = forecast_horizon,
            forecast_vol_ann  = fwd_vol,
            forecast_ci_lo    = pd.Series(ci_lo.values, index=fwd_dates),
            forecast_ci_hi    = pd.Series(ci_hi.values, index=fwd_dates),
            vol_regime        = _vol_regime(pct_rank),
            vol_pct_rank      = pct_rank,
            garch_omega       = omega,
            garch_alpha       = alpha,
            garch_beta        = beta,
            garch_hl          = float(hl),
        )

    except Exception as exc:
        logger.warning("GARCH fitting failed (%s); falling back to EWMA.", exc)
        res = ewma_vol(prices, forecast_horizon=forecast_horizon)
        res.method = "GARCH(1,1) [EWMA fallback]"
        return res


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vol_regime(pct_rank: float) -> str:
    if pct_rank < 0.25:
        return "Low"
    elif pct_rank < 0.60:
        return "Normal"
    elif pct_rank < 0.85:
        return "Elevated"
    else:
        return "High"


VOL_REGIME_COLOURS = {
    "Low":      "#00D4FF",
    "Normal":   "#00C864",
    "Elevated": "#FFD700",
    "High":     "#FF4444",
}
