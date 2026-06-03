"""
AlphaForge — Forecast Validation Engine
=========================================
Walk-forward out-of-sample evaluation of the GBM probabilistic forecasting
model.  Asks: "How well has this model's probability distribution captured
realised outcomes in the past?"

Methodology
-----------
Walk-forward protocol:
  1. Train on [t - train_len, t)
  2. Generate n_step-ahead GBM forecast
  3. Compare median forecast to realised price
  4. Evaluate calibration (does the 95% CI contain the outcome 95% of the time?)
  5. Slide window forward; repeat

Metrics
-------
  MAE                  Mean Absolute Error  (median forecast vs. realised)
  RMSE                 Root Mean Squared Error
  MAPE                 Mean Absolute Percentage Error
  Directional Accuracy  Sign of Δ correct?
  Hit Rate             P(realised within 95% CI band)
  Calibration Error    |Nominal coverage - Empirical coverage|
  PIT Uniformity       Probability Integral Transform test (U[0,1] expected)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .monte_carlo import run_gbm_simulation

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Validation metrics are based on historical walk-forward evaluation. "
    "Past forecast performance does not guarantee future accuracy."
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ticker:      str
    train_len:   int
    step:        int       # forecast horizon used
    n_windows:   int

    # Point forecast metrics (median vs. realised)
    mae:     float
    rmse:    float
    mape:    float
    dir_acc: float    # directional accuracy

    # Probabilistic metrics
    hit_rate_50:  float   # coverage of 50% CI
    hit_rate_68:  float
    hit_rate_90:  float
    hit_rate_95:  float
    calibration_error: float   # mean |nominal - empirical| across levels

    # PIT test
    pit_values:   np.ndarray   # should be U[0,1] if model is calibrated
    pit_p_value:  float        # KS test p-value (>0.05 = can't reject uniformity)

    # Time series of forecasts vs. actuals
    forecast_dates:  pd.DatetimeIndex
    median_forecasts: pd.Series
    realised:         pd.Series
    errors:           pd.Series

    disclaimer: str = field(default=_DISCLAIMER, repr=False)

    def grade(self) -> str:
        """Qualitative calibration grade based on hit rates and PIT."""
        score = 0
        if abs(self.hit_rate_95 - 0.95) < 0.08: score += 2
        if abs(self.hit_rate_68 - 0.68) < 0.08: score += 2
        if self.dir_acc > 0.52:                  score += 1
        if self.pit_p_value > 0.05:              score += 2
        if self.calibration_error < 0.05:        score += 2
        if score >= 7:  return "Excellent"
        if score >= 5:  return "Good"
        if score >= 3:  return "Fair"
        return "Poor"

    def summary_df(self) -> pd.DataFrame:
        rows = [
            ("MAE",                     f"${self.mae:,.2f}"),
            ("RMSE",                    f"${self.rmse:,.2f}"),
            ("MAPE",                    f"{self.mape*100:.2f}%"),
            ("Directional Accuracy",    f"{self.dir_acc*100:.1f}%"),
            ("Hit Rate — 50% CI",       f"{self.hit_rate_50*100:.1f}%  (ideal 50%)"),
            ("Hit Rate — 68% CI",       f"{self.hit_rate_68*100:.1f}%  (ideal 68%)"),
            ("Hit Rate — 90% CI",       f"{self.hit_rate_90*100:.1f}%  (ideal 90%)"),
            ("Hit Rate — 95% CI",       f"{self.hit_rate_95*100:.1f}%  (ideal 95%)"),
            ("Calibration Error",       f"{self.calibration_error*100:.2f}%"),
            ("PIT KS p-value",          f"{self.pit_p_value:.4f}"),
            ("Windows Evaluated",       str(self.n_windows)),
            ("Forecast Grade",          self.grade()),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


# ── Walk-forward engine ───────────────────────────────────────────────────────

def run_walk_forward_validation(
    prices: pd.Series,
    train_len: int = 252,
    step: int = 21,
    n_paths: int = 1_000,
    ticker: str = "ASSET",
    seed: int = 42,
) -> ValidationResult:
    """
    Walk-forward validation of the GBM forecasting model.

    Parameters
    ----------
    prices    : historical adjusted-close price series
    train_len : number of bars used to calibrate each window's GBM params
    step      : forecast horizon (bars ahead) evaluated per window
    n_paths   : MC paths per window
    ticker    : label
    seed      : RNG seed

    Returns
    -------
    ValidationResult
    """
    n = len(prices)
    min_bars = train_len + step
    if n < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for walk-forward validation "
            f"(train_len={train_len}, step={step})."
        )

    rng    = np.random.default_rng(seed)
    starts = range(train_len, n - step, max(step, 5))

    median_preds: list[float] = []
    realised_vals: list[float] = []
    forecast_dates_list: list[pd.Timestamp] = []
    pct_lo_50: list[float] = []; pct_hi_50: list[float] = []
    pct_lo_68: list[float] = []; pct_hi_68: list[float] = []
    pct_lo_90: list[float] = []; pct_hi_90: list[float] = []
    pct_lo_95: list[float] = []; pct_hi_95: list[float] = []
    pit_vals: list[float] = []

    for t in starts:
        train  = prices.iloc[t - train_len: t]
        target_price = float(prices.iloc[t + step - 1])
        win_seed = int(rng.integers(0, 2**31))

        try:
            mc = run_gbm_simulation(train, horizon=step, n_paths=n_paths,
                                    ticker=ticker, seed=win_seed)
        except Exception:
            continue

        terminal = mc.terminal_values
        median_p = float(np.median(terminal))
        median_preds.append(median_p)
        realised_vals.append(target_price)
        forecast_dates_list.append(prices.index[t + step - 1])

        # CI bands for terminal value
        for lo_p, hi_p, lo_l, hi_l in [
            (25, 75, pct_lo_50, pct_hi_50),
            (16, 84, pct_lo_68, pct_hi_68),
            (5,  95, pct_lo_90, pct_hi_90),
            (2.5,97.5,pct_lo_95,pct_hi_95),
        ]:
            lo_l.append(float(np.percentile(terminal, lo_p)))
            hi_l.append(float(np.percentile(terminal, hi_p)))

        # PIT: empirical CDF of terminal at realised price
        pit_vals.append(float(np.mean(terminal <= target_price)))

    if len(median_preds) < 5:
        raise ValueError("Insufficient validation windows. Reduce train_len or step.")

    median_arr   = np.array(median_preds)
    realised_arr = np.array(realised_vals)
    errors_arr   = median_arr - realised_arr
    fd_idx       = pd.DatetimeIndex(forecast_dates_list)

    # ── Point metrics ──────────────────────────────────────────────────────────
    mae  = float(np.mean(np.abs(errors_arr)))
    rmse = float(np.sqrt(np.mean(errors_arr ** 2)))
    mape = float(np.mean(np.abs(errors_arr / realised_arr)))

    price_changes_median = np.diff(median_arr)
    price_changes_real   = np.diff(realised_arr)
    dir_acc = float(
        np.mean(np.sign(price_changes_median) == np.sign(price_changes_real))
    ) if len(price_changes_median) > 0 else 0.5

    # ── Coverage / hit rates ───────────────────────────────────────────────────
    def _hit(lo_list: list[float], hi_list: list[float]) -> float:
        lo = np.array(lo_list); hi = np.array(hi_list)
        inside = (realised_arr >= lo) & (realised_arr <= hi)
        return float(np.mean(inside))

    hr50 = _hit(pct_lo_50, pct_hi_50)
    hr68 = _hit(pct_lo_68, pct_hi_68)
    hr90 = _hit(pct_lo_90, pct_hi_90)
    hr95 = _hit(pct_lo_95, pct_hi_95)

    cal_err = float(np.mean([
        abs(hr50 - 0.50), abs(hr68 - 0.68),
        abs(hr90 - 0.90), abs(hr95 - 0.95),
    ]))

    # ── PIT uniformity test ────────────────────────────────────────────────────
    pit_arr = np.array(pit_vals)
    _, pit_p = stats.kstest(pit_arr, "uniform")

    return ValidationResult(
        ticker           = ticker,
        train_len        = train_len,
        step             = step,
        n_windows        = len(median_preds),
        mae              = mae,
        rmse             = rmse,
        mape             = mape,
        dir_acc          = dir_acc,
        hit_rate_50      = hr50,
        hit_rate_68      = hr68,
        hit_rate_90      = hr90,
        hit_rate_95      = hr95,
        calibration_error= cal_err,
        pit_values       = pit_arr,
        pit_p_value      = float(pit_p),
        forecast_dates   = fd_idx,
        median_forecasts = pd.Series(median_arr, index=fd_idx),
        realised         = pd.Series(realised_arr, index=fd_idx),
        errors           = pd.Series(errors_arr, index=fd_idx),
    )
