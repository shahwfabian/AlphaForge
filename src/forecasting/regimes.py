"""
AlphaForge — Market Regime Detection Engine
============================================
Implements a Gaussian Hidden Markov Model entirely from scratch using
the Baum-Welch EM algorithm and the Viterbi MAP-decoding algorithm.
No hmmlearn dependency — uses only numpy + scipy.

Mathematical framework
----------------------
State space: K hidden states (2 or 3)
Observations: O_t = [r_t, σ̂_t]  (log-return, rolling vol)

Emission:   b_k(O_t) = N(O_t | μ_k, Σ_k)   multivariate Gaussian
Transition: P(q_t = j | q_{t-1} = i) = A_{ij}   row-stochastic
Initial:    P(q_1 = k) = π_k

Estimation: Baum-Welch (EM)
Decoding:   Viterbi (MAP state sequence)
Smoothing:  Forward-Backward (posterior state probabilities)

All operations in log-space to prevent underflow.

Regime labelling
----------------
States are sorted by mean return μ_k[0]:
  K=2: [Bear/Risk-Off, Bull/Risk-On]
  K=3: [Bear, Neutral, Bull]
  Volatility characterisation overlaid by comparing σ_k to global median.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Regime classifications are statistical inferences from historical "
    "return and volatility patterns.  They do not predict future regimes "
    "and should not be used as the sole basis for investment decisions."
)

# ── Regime label maps ──────────────────────────────────────────────────────────

_LABELS_2 = {0: "Bear / Risk-Off", 1: "Bull / Risk-On"}
_LABELS_3 = {0: "Bear / High Vol", 1: "Neutral", 2: "Bull / Low Vol"}

_COLOURS_2 = {0: "#FF4444", 1: "#00D4FF"}
_COLOURS_3 = {0: "#FF4444", 1: "#FFD700", 2: "#00D4FF"}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    n_states: int
    state_sequence: np.ndarray          # most-likely regime at each bar (Viterbi)
    state_probs: np.ndarray             # (T, K) smoothed posterior probabilities
    current_state: int                  # most recent Viterbi state
    current_probs: np.ndarray           # (K,) probabilities for the last bar

    transition_matrix: np.ndarray       # (K, K) estimated A
    stationary_dist: np.ndarray         # (K,) stationary distribution of A

    regime_means: np.ndarray            # (K, 2) [mean_return, mean_vol]
    regime_stds: np.ndarray             # (K, 2) [std_return, std_vol]

    dates: pd.DatetimeIndex
    prices: pd.Series                   # original price series for overlay

    labels: dict[int, str]              # {state_idx: label_str}
    colours: dict[int, str]

    log_likelihood: float
    n_iter: int                         # EM iterations until convergence

    disclaimer: str = field(default=_DISCLAIMER, repr=False)

    @property
    def current_label(self) -> str:
        return self.labels[self.current_state]

    @property
    def current_colour(self) -> str:
        return self.colours[self.current_state]

    def regime_persistence(self) -> dict[str, float]:
        """P(stay in same regime) for each state."""
        return {
            self.labels[k]: float(self.transition_matrix[k, k])
            for k in range(self.n_states)
        }

    def transition_df(self) -> pd.DataFrame:
        """Human-readable transition probability table."""
        lbls = [self.labels[k] for k in range(self.n_states)]
        return pd.DataFrame(
            self.transition_matrix,
            index=pd.Index(lbls, name="From"),
            columns=pd.Index(lbls, name="To"),
        )

    def regime_stats_df(self) -> pd.DataFrame:
        rows = []
        for k in range(self.n_states):
            rows.append({
                "Regime":           self.labels[k],
                "Mean Return (ann)":f"{self.regime_means[k, 0] * 252 * 100:+.1f}%",
                "Volatility (ann)": f"{self.regime_stds[k, 0] * np.sqrt(252) * 100:.1f}%",
                "Stationary Prob":  f"{self.stationary_dist[k]*100:.1f}%",
                "Persistence":      f"{self.transition_matrix[k, k]*100:.1f}%",
            })
        return pd.DataFrame(rows)


# ── Gaussian HMM (Baum-Welch + Viterbi) ──────────────────────────────────────

class _GaussianHMM:
    """
    2-or-3-state Gaussian HMM estimated via Baum-Welch EM.
    Numerical stability via log-sum-exp throughout.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 150, tol: float = 1e-4,
                 random_state: int = 42):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        # Parameters – initialised in fit()
        self.log_pi: np.ndarray = None      # (K,)
        self.log_A:  np.ndarray = None      # (K, K) log transition
        self.means:  np.ndarray = None      # (K, D)
        self.covars: list[np.ndarray] = None  # list of K (D, D) cov matrices

        self.log_likelihood_: float = -np.inf
        self.n_iter_: int = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_params(self, X: np.ndarray):
        """Initialise with equal transition, K-percentile means, diagonal covars."""
        T, D = X.shape
        K = self.K

        # Uniform initial + transition
        self.log_pi = np.log(np.full(K, 1.0 / K))
        A = np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * 0.9
        A /= A.sum(axis=1, keepdims=True)
        self.log_A = np.log(A + 1e-300)

        # Initialise means at K evenly spaced percentiles of the first feature
        pcts = np.linspace(10, 90, K)
        indices = [np.argmin(np.abs(X[:, 0] - np.percentile(X[:, 0], p)))
                   for p in pcts]
        self.means  = X[indices].astype(float)
        global_cov  = np.cov(X.T) + np.eye(D) * 1e-6
        self.covars = [global_cov.copy() for _ in range(K)]

    # ── Log-emission ──────────────────────────────────────────────────────────

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) log-emission matrix."""
        T = len(X)
        log_b = np.zeros((T, self.K))
        for k in range(self.K):
            cov_k = self.covars[k] + np.eye(X.shape[1]) * 1e-8
            try:
                log_b[:, k] = multivariate_normal.logpdf(X, self.means[k], cov_k)
            except Exception:
                log_b[:, k] = -1e10
        return log_b

    # ── Forward (log-space) ───────────────────────────────────────────────────

    def _forward(self, log_b: np.ndarray):
        T, K = log_b.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + self.log_A[:, k]) + log_b[t, k]
        return log_alpha

    # ── Backward (log-space) ──────────────────────────────────────────────────

    def _backward(self, log_b: np.ndarray):
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        # log_beta[T-1] = 0 (log 1)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(
                    self.log_A[k] + log_b[t + 1] + log_beta[t + 1]
                )
        return log_beta

    # ── E-step ────────────────────────────────────────────────────────────────

    def _estep(self, X: np.ndarray):
        T, K = len(X), self.K
        log_b     = self._log_emission(X)
        log_alpha = self._forward(log_b)
        log_beta  = self._backward(log_b)

        # Log-likelihood
        log_lik = logsumexp(log_alpha[-1])

        # Gamma: (T, K) posterior state probabilities
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi: (T-1, K, K) joint transition posteriors
        log_xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    log_xi[t, j, k] = (
                        log_alpha[t, j]
                        + self.log_A[j, k]
                        + log_b[t + 1, k]
                        + log_beta[t + 1, k]
                    )
            log_xi[t] -= logsumexp(log_xi[t].ravel())
        xi = np.exp(log_xi)

        return gamma, xi, log_lik

    # ── M-step ────────────────────────────────────────────────────────────────

    def _mstep(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        T, D = X.shape
        K = self.K

        # Initial state
        self.log_pi = np.log(gamma[0] + 1e-300)

        # Transition matrix
        A = xi.sum(axis=0) + 1e-10                # (K, K)
        A /= A.sum(axis=1, keepdims=True)
        self.log_A = np.log(A + 1e-300)

        # Means and covariances
        g_sum = gamma.sum(axis=0) + 1e-10         # (K,)
        for k in range(K):
            w = gamma[:, k] / g_sum[k]
            self.means[k] = (w[:, None] * X).sum(axis=0)
            diff = X - self.means[k]
            self.covars[k] = (w[:, None] * diff).T @ diff + np.eye(D) * 1e-6

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "_GaussianHMM":
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        prev_ll = -np.inf
        for i in range(self.n_iter):
            gamma, xi, log_lik = self._estep(X)
            self._mstep(X, gamma, xi)
            if abs(log_lik - prev_ll) < self.tol:
                self.n_iter_ = i + 1
                break
            prev_ll = log_lik
        else:
            self.n_iter_ = self.n_iter
        self.log_likelihood_ = float(prev_ll)
        return self

    # ── Predict (Viterbi) ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        T, K = len(X), self.K
        log_b = self._log_emission(X)
        delta = np.zeros((T, K))
        psi   = np.zeros((T, K), dtype=int)

        delta[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            for k in range(K):
                scores      = delta[t - 1] + self.log_A[:, k]
                psi[t, k]   = int(np.argmax(scores))
                delta[t, k] = scores[psi[t, k]] + log_b[t, k]

        # Traceback
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    # ── Smoothed probabilities ────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_b     = self._log_emission(X)
        log_alpha = self._forward(log_b)
        log_beta  = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)


# ── Stationary distribution ───────────────────────────────────────────────────

def _stationary(A: np.ndarray) -> np.ndarray:
    """Compute stationary distribution of a row-stochastic transition matrix."""
    K = A.shape[0]
    # Solve π = π A  ↔  (A^T - I) π = 0,  Σπ = 1
    M = (A.T - np.eye(K))
    M = np.vstack([M, np.ones(K)])
    b = np.zeros(K + 1); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(M, b, rcond=None)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


# ── Public API ────────────────────────────────────────────────────────────────

def detect_regimes(
    prices: pd.Series,
    n_states: int = 3,
    window: int = 20,
    n_iter: int = 150,
    seed: int = 42,
) -> RegimeResult:
    """
    Fit a Gaussian HMM to classify historical market regimes.

    Observations: [log-return, rolling annualised volatility]

    Parameters
    ----------
    prices    : pd.Series of adjusted close prices
    n_states  : 2 or 3 hidden states
    window    : rolling window for vol estimation (default 20 days)
    n_iter    : maximum Baum-Welch EM iterations
    seed      : RNG seed

    Returns
    -------
    RegimeResult
    """
    if len(prices) < max(60, window * 3):
        raise ValueError("Need at least 60 bars for regime detection.")
    if n_states not in (2, 3):
        raise ValueError("n_states must be 2 or 3.")

    # ── Feature engineering ────────────────────────────────────────────────────
    log_ret = np.log(prices / prices.shift(1)).dropna()
    roll_vol = log_ret.rolling(window).std().dropna() * np.sqrt(252)

    # Align
    aligned = pd.concat([log_ret, roll_vol], axis=1).dropna()
    aligned.columns = ["ret", "vol"]
    X = aligned.values   # (T, 2)

    # ── Fit HMM ────────────────────────────────────────────────────────────────
    model = _GaussianHMM(n_states=n_states, n_iter=n_iter, tol=1e-4,
                         random_state=seed)
    model.fit(X)

    states      = model.predict(X)          # Viterbi sequence
    smoothed    = model.predict_proba(X)    # (T, K) posterior probs

    A     = np.exp(model.log_A)
    A    /= A.sum(axis=1, keepdims=True)    # re-normalise after exp

    pi_stat = _stationary(A)

    # ── Sort states by mean return (ascending: Bear → Bull) ───────────────────
    mean_rets = model.means[:, 0]
    order     = np.argsort(mean_rets)      # ascending → index 0 = Bear
    inv_order = np.argsort(order)

    sorted_states = inv_order[states]
    sorted_probs  = smoothed[:, order]
    sorted_A      = A[np.ix_(order, order)]
    sorted_means  = model.means[order]
    sorted_stds   = np.stack([
        np.sqrt(np.diag(model.covars[k])) for k in order
    ])
    sorted_pi     = pi_stat[order]

    labels  = _LABELS_2 if n_states == 2 else _LABELS_3
    colours = _COLOURS_2 if n_states == 2 else _COLOURS_3

    dates_aligned = aligned.index

    return RegimeResult(
        n_states          = n_states,
        state_sequence    = sorted_states,
        state_probs       = sorted_probs,
        current_state     = int(sorted_states[-1]),
        current_probs     = sorted_probs[-1],
        transition_matrix = sorted_A,
        stationary_dist   = sorted_pi,
        regime_means      = sorted_means,
        regime_stds       = sorted_stds,
        dates             = dates_aligned,
        prices            = prices.loc[dates_aligned],
        labels            = labels,
        colours           = colours,
        log_likelihood    = model.log_likelihood_,
        n_iter            = model.n_iter_,
    )
