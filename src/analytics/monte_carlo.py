"""
Monte Carlo portfolio stress testing.

Simulation methods
------------------
1. **Gaussian IID**        — draw from N(μ, σ). Fast. Ignores fat tails.
2. **Block Bootstrap**     — resample overlapping blocks of historical returns.
                             Preserves autocorrelation and non-normality.
3. **GBM (log-normal)**    — Geometric Brownian Motion; standard options-theory model.
4. **Stress Scenarios**    — historical crisis replay using actual daily return sequences.
5. **Cholesky Multi-asset**— correlated multi-asset simulation via Cholesky factorisation.

NOTE: All results are illustrative research tools, not predictions.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# ── Historical stress scenarios ───────────────────────────────────────────────
# Approximate daily returns for selected crisis periods.
# Using SPY-like drawdown magnitudes and durations.
STRESS_SCENARIOS: dict[str, dict] = {
    "2008 Global Financial Crisis": {
        "description": "Peak drawdown Oct 2007 – Mar 2009 (~−57%)",
        "daily_mu":   -0.00220,
        "daily_sigma": 0.02800,
        "duration_days": 375,
    },
    "COVID-19 Crash (Feb–Mar 2020)": {
        "description": "Fastest 30%+ drawdown in history (~33 days)",
        "daily_mu":   -0.01050,
        "daily_sigma": 0.04200,
        "duration_days": 33,
    },
    "2000 Dot-Com Bust": {
        "description": "Nasdaq −78% over ~30 months",
        "daily_mu":   -0.00180,
        "daily_sigma": 0.02100,
        "duration_days": 504,
    },
    "2022 Rate-Hike Bear Market": {
        "description": "S&P 500 −27%, bonds −18% simultaneously",
        "daily_mu":   -0.00120,
        "daily_sigma": 0.01600,
        "duration_days": 252,
    },
}


# ── 1. Gaussian IID simulation ────────────────────────────────────────────────

def run_monte_carlo(
    returns: pd.Series,
    initial_value: float = 100_000.0,
    n_simulations: int = 500,
    n_days: int = 252,
    random_seed: int = 42,
) -> dict:
    """
    Gaussian IID Monte Carlo simulation.

    Draws daily returns from N(μ, σ) where μ and σ are estimated from
    the historical return series.

    Parameters
    ----------
    returns : pd.Series
        Historical daily return series.
    initial_value : float
    n_simulations : int
    n_days : int
    random_seed : int

    Returns
    -------
    dict  (keys: paths, final_values, percentiles, prob_loss, mu, sigma, …)
    """
    clean = returns.dropna()
    if len(clean) < 20:
        logger.error("Insufficient return history for Monte Carlo.")
        return {}

    mu    = float(clean.mean())
    sigma = float(clean.std())
    rng   = np.random.default_rng(random_seed)

    sim_returns = rng.normal(loc=mu, scale=sigma, size=(n_simulations, n_days))
    paths       = initial_value * np.cumprod(1 + sim_returns, axis=1)
    final       = paths[:, -1]

    return _build_result(
        paths, final, initial_value, n_simulations, n_days,
        mu, sigma, method="Gaussian IID",
    )


# ── 2. Block bootstrap ────────────────────────────────────────────────────────

def block_bootstrap_mc(
    returns: pd.Series,
    initial_value: float = 100_000.0,
    n_simulations: int = 500,
    n_days: int = 252,
    block_size: int = 21,
    random_seed: int = 42,
) -> dict:
    """
    Block bootstrap Monte Carlo.

    Resamples overlapping blocks of ``block_size`` consecutive historical
    returns instead of drawing i.i.d.  This preserves:
    - Serial autocorrelation (momentum / mean-reversion effects).
    - Fat tails and skewness of the empirical distribution.
    - Volatility clustering (GARCH-like behaviour).

    Parameters
    ----------
    returns : pd.Series
    initial_value : float
    n_simulations : int
    n_days : int
    block_size : int
        Length of each resampled block in trading days.
    random_seed : int

    Returns
    -------
    dict
    """
    clean = returns.dropna().values
    n_hist = len(clean)
    if n_hist < block_size * 2:
        logger.warning("History too short for block bootstrap. Falling back to IID.")
        return run_monte_carlo(
            returns, initial_value, n_simulations, n_days, random_seed
        )

    rng = np.random.default_rng(random_seed)
    n_blocks = int(np.ceil(n_days / block_size))

    # All valid block start indices
    valid_starts = np.arange(0, n_hist - block_size + 1)
    paths = np.zeros((n_simulations, n_days))

    for sim in range(n_simulations):
        # Draw block starts randomly with replacement
        starts = rng.choice(valid_starts, size=n_blocks, replace=True)
        sampled = np.concatenate([clean[s: s + block_size] for s in starts])
        sampled = sampled[:n_days]  # trim to exact length
        paths[sim] = initial_value * np.cumprod(1 + sampled)

    final = paths[:, -1]
    mu, sigma = float(clean.mean()), float(clean.std())

    return _build_result(
        paths, final, initial_value, n_simulations, n_days,
        mu, sigma, method="Block Bootstrap",
        extra={"block_size": block_size},
    )


# ── 3. GBM (log-normal) simulation ───────────────────────────────────────────

def gbm_monte_carlo(
    returns: pd.Series,
    initial_value: float = 100_000.0,
    n_simulations: int = 500,
    n_days: int = 252,
    random_seed: int = 42,
) -> dict:
    """
    Geometric Brownian Motion (GBM) simulation.

    Uses log-return statistics:
        log(S_t / S_{t-1}) ~ N(μ_log - σ²/2, σ_log)

    The drift correction (Itô correction) ensures that the expected
    simple return equals the arithmetic mean, not the geometric mean.

    Parameters
    ----------
    returns : pd.Series
    initial_value : float
    n_simulations : int
    n_days : int
    random_seed : int

    Returns
    -------
    dict
    """
    clean = returns.dropna()
    log_rets = np.log(1 + clean)
    mu_log   = float(log_rets.mean())
    sig_log  = float(log_rets.std())

    # Itô drift correction
    drift  = mu_log - 0.5 * sig_log ** 2
    rng    = np.random.default_rng(random_seed)

    log_shocks   = rng.normal(loc=drift, scale=sig_log, size=(n_simulations, n_days))
    cum_log_rets = np.cumsum(log_shocks, axis=1)
    paths        = initial_value * np.exp(cum_log_rets)
    final        = paths[:, -1]

    return _build_result(
        paths, final, initial_value, n_simulations, n_days,
        float(clean.mean()), float(clean.std()), method="GBM Log-Normal",
    )


# ── 4. Stress scenario simulation ────────────────────────────────────────────

def stress_scenario_mc(
    initial_value: float = 100_000.0,
    scenario_name: str = "2008 Global Financial Crisis",
    n_simulations: int = 500,
    random_seed: int = 42,
) -> dict:
    """
    Simulate portfolio paths under a historical stress scenario.

    Each simulation draws from the scenario's return distribution
    (calibrated to the actual crisis statistics).

    Parameters
    ----------
    initial_value : float
    scenario_name : str
        One of the keys in ``STRESS_SCENARIOS``.
    n_simulations : int
    random_seed : int

    Returns
    -------
    dict  (includes 'scenario' key with description)
    """
    if scenario_name not in STRESS_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(STRESS_SCENARIOS.keys())}"
        )

    scn   = STRESS_SCENARIOS[scenario_name]
    mu    = scn["daily_mu"]
    sigma = scn["daily_sigma"]
    n_days = scn["duration_days"]

    rng         = np.random.default_rng(random_seed)
    sim_returns = rng.normal(loc=mu, scale=sigma, size=(n_simulations, n_days))
    paths       = initial_value * np.cumprod(1 + sim_returns, axis=1)
    final       = paths[:, -1]

    result = _build_result(
        paths, final, initial_value, n_simulations, n_days,
        mu, sigma, method="Stress Scenario",
    )
    result["scenario_name"]        = scenario_name
    result["scenario_description"] = scn["description"]
    return result


# ── 5. Multi-asset Cholesky simulation ────────────────────────────────────────

def cholesky_mc(
    returns_matrix: pd.DataFrame,
    initial_values: dict[str, float],
    n_simulations: int = 300,
    n_days: int = 252,
    random_seed: int = 42,
) -> dict:
    """
    Correlated multi-asset Monte Carlo via Cholesky factorisation.

    Generates correlated return paths that honour the empirical
    correlation structure of the asset universe.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Columns = tickers, rows = daily returns (aligned).
    initial_values : dict[ticker → initial portfolio value]
    n_simulations : int
    n_days : int
    random_seed : int

    Returns
    -------
    dict  (per-asset paths and summary stats)
    """
    clean = returns_matrix.dropna()
    if len(clean) < 30:
        return {}

    tickers = clean.columns.tolist()
    mu_vec  = clean.mean().values        # (n_assets,)
    cov_mat = clean.cov().values         # (n_assets, n_assets)

    try:
        L = np.linalg.cholesky(cov_mat + 1e-8 * np.eye(len(tickers)))
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix not positive definite. Adding jitter.")
        cov_mat += 1e-6 * np.eye(len(tickers))
        L = np.linalg.cholesky(cov_mat)

    rng = np.random.default_rng(random_seed)
    # Shape: (n_simulations, n_days, n_assets)
    z     = rng.standard_normal((n_simulations, n_days, len(tickers)))
    corr_z = z @ L.T + mu_vec           # apply Cholesky + mean drift

    results_per_asset = {}
    for i, ticker in enumerate(tickers):
        iv = initial_values.get(ticker, 100_000.0)
        asset_rets = corr_z[:, :, i]                    # (n_sim, n_days)
        paths      = iv * np.cumprod(1 + asset_rets, axis=1)
        final      = paths[:, -1]
        results_per_asset[ticker] = {
            "paths":       paths,
            "final_values": final,
            "percentile_5":  float(np.percentile(final, 5)),
            "percentile_50": float(np.percentile(final, 50)),
            "percentile_95": float(np.percentile(final, 95)),
            "prob_loss":     float(np.mean(final < iv)),
        }

    return {
        "assets": results_per_asset,
        "tickers": tickers,
        "correlation_matrix": pd.DataFrame(
            clean.corr().values, index=tickers, columns=tickers
        ),
        "n_simulations": n_simulations,
        "n_days": n_days,
        "method": "Cholesky Multi-Asset",
    }


# ── Shared result builder ─────────────────────────────────────────────────────

def _build_result(
    paths: np.ndarray,
    final: np.ndarray,
    initial_value: float,
    n_simulations: int,
    n_days: int,
    mu: float,
    sigma: float,
    method: str = "Unknown",
    extra: dict | None = None,
) -> dict:
    """Construct the standardised Monte Carlo result dict."""
    result = {
        "paths":           paths,
        "final_values":    final,
        "percentile_5":    float(np.percentile(final, 5)),
        "percentile_25":   float(np.percentile(final, 25)),
        "percentile_50":   float(np.percentile(final, 50)),
        "percentile_75":   float(np.percentile(final, 75)),
        "percentile_95":   float(np.percentile(final, 95)),
        "mean_final":      float(np.mean(final)),
        "std_final":       float(np.std(final)),
        "prob_loss":       float(np.mean(final < initial_value)),
        "prob_gain_20pct": float(np.mean(final > initial_value * 1.20)),
        "var_5pct":        float(np.percentile(final / initial_value - 1, 5)),
        "mu":              mu,
        "sigma":           sigma,
        "n_simulations":   n_simulations,
        "n_days":          n_days,
        "initial_value":   initial_value,
        "method":          method,
    }
    if extra:
        result.update(extra)
    return result


# ── Percentile DataFrame for plotting ────────────────────────────────────────

def mc_percentile_df(mc_results: dict) -> pd.DataFrame:
    """Build a tidy percentile-band DataFrame for fan-chart plotting."""
    if not mc_results or "paths" not in mc_results:
        return pd.DataFrame()

    paths  = mc_results["paths"]
    n_days = paths.shape[1]
    days   = np.arange(1, n_days + 1)

    return pd.DataFrame({
        "Day": days,
        "P5":  np.percentile(paths, 5,  axis=0),
        "P25": np.percentile(paths, 25, axis=0),
        "P50": np.percentile(paths, 50, axis=0),
        "P75": np.percentile(paths, 75, axis=0),
        "P95": np.percentile(paths, 95, axis=0),
    })
