"""
hmm_engine.py — HMM regime detection engine.

DESIGN PHILOSOPHY
─────────────────
The HMM is a VOLATILITY CLASSIFIER.  It identifies whether the market is in a
calm, moderate, or turbulent environment.  It does NOT predict price direction.
The strategy layer uses the volatility regime to set portfolio allocation:
  - be fully invested when conditions are calm
  - reduce exposure when turbulent

NO LOOK-AHEAD BIAS
──────────────────
All inference uses the forward algorithm (filtered inference) ONLY.
model.predict() (Viterbi) is NEVER used for live or backtest decisions.
Viterbi processes the entire sequence at once and revises past states using
future observations — this is look-ahead bias that inflates backtest metrics.

The forward algorithm computes P(state_t | obs_{0:t}) using only observations
up to and including t.  Changing obs[t+1:] has zero effect on P(state_t | …).

LABEL SYSTEM
────────────
States are labelled in two independent ways:
  1. RegimeLabel  — sorted by MEAN RETURN (ascending) for human readability
  2. Regime       — sorted by VOLATILITY for use by the strategy layer
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Return-sorted label maps  (index 0 = most bearish → most bullish)
# ─────────────────────────────────────────────────────────────────────────────
_LABEL_MAP: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: [
        "CRASH",
        "STRONG_BEAR",
        "WEAK_BEAR",
        "NEUTRAL",
        "WEAK_BULL",
        "STRONG_BULL",
        "EUPHORIA",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class RegimeLabel(str, Enum):
    """Return-based regime labels (sorted ascending by expected return).
    Used for human-readable logging and reporting only."""

    CRASH = "CRASH"
    STRONG_BEAR = "STRONG_BEAR"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    WEAK_BULL = "WEAK_BULL"
    BULL = "BULL"
    STRONG_BULL = "STRONG_BULL"
    EUPHORIA = "EUPHORIA"
    UNKNOWN = "UNKNOWN"


class Regime(str, Enum):
    """Volatility-based regime classification consumed by the strategy layer."""

    LOW_VOL = "low_vol"
    MID_VOL = "mid_vol"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    """Static per-state metadata populated after fitting."""

    regime_id: int
    regime_name: str                   # RegimeLabel.value
    expected_return: float             # annualised (× 252) from HMM means feature-0
    expected_volatility: float         # annualised (× √252) from covariance diagonal
    recommended_strategy_type: str     # 'full_invest' | 'reduced' | 'defensive'
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Full state snapshot at a single bar — primary output of HMMEngine."""

    label: RegimeLabel                 # return-based label (human-readable)
    regime: Regime                     # volatility-based class (strategy input)
    state_id: int                      # raw HMM state index
    probability: float                 # posterior P(state_id | obs_{0:t})
    state_probabilities: np.ndarray    # full posterior vector, shape (n_states,)
    timestamp: Optional[pd.Timestamp]
    is_confirmed: bool                 # True after stability_bars consecutive bars
    consecutive_bars: int              # bars in current confirmed regime


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible container
# ─────────────────────────────────────────────────────────────────────────────

class RegimeResult:
    """
    Backward-compatible output container.

    Keeps the original interface used by test files and the strategy / signal
    layers, while also carrying the new fields introduced in RegimeState.
    """

    def __init__(
        self,
        regime: Regime,
        state_index: int,
        confidence: float,
        posteriors: np.ndarray,
        n_states: int,
        label: RegimeLabel = RegimeLabel.UNKNOWN,
        is_confirmed: bool = False,
        consecutive_bars: int = 1,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        self.regime = regime
        self.state_index = state_index
        self.confidence = confidence
        self.posteriors = posteriors
        self.n_states = n_states
        self.label = label
        self.is_confirmed = is_confirmed
        self.consecutive_bars = consecutive_bars
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (
            f"RegimeResult(regime={self.regime}, label={self.label}, "
            f"state={self.state_index}, confidence={self.confidence:.3f}, "
            f"confirmed={self.is_confirmed})"
        )

    @classmethod
    def from_regime_state(cls, rs: RegimeState, n_states: int) -> "RegimeResult":
        """Wrap a RegimeState in a RegimeResult for backward compatibility."""
        return cls(
            regime=rs.regime,
            state_index=rs.state_id,
            confidence=rs.probability,
            posteriors=rs.state_probabilities,
            n_states=n_states,
            label=rs.label,
            is_confirmed=rs.is_confirmed,
            consecutive_bars=rs.consecutive_bars,
            timestamp=rs.timestamp,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class HMMEngine:
    """
    Gaussian HMM regime detection engine with automatic model selection.

    Parameters
    ----------
    n_candidates : list[int]
        Candidate state counts for BIC model selection (default [3, 4, 5, 6, 7]).
    n_init : int
        Random restarts per candidate to escape local optima (default 10).
    covariance_type : str
        HMM covariance — 'full' | 'diag' | 'spherical' | 'tied' (default 'full').
    min_train_bars : int
        Minimum valid feature rows required to fit (default 504 ≈ 2 years).
    stability_bars : int
        Consecutive identical bars before a regime change is confirmed (default 3).
    flicker_window : int
        Rolling window for counting regime transitions (default 20).
    flicker_threshold : int
        Transitions in flicker_window that trigger uncertainty mode (default 4).
    min_confidence : float
        Minimum posterior probability to emit a non-UNKNOWN signal (default 0.55).
    """

    def __init__(
        self,
        n_candidates: Optional[list[int]] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        min_train_bars: int = 504,
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
    ) -> None:
        self.n_candidates: list[int] = n_candidates or [3, 4, 5, 6, 7]
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_train_bars = min_train_bars
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence

        # Populated by fit()
        self._model: Optional[GaussianHMM] = None
        self._n_states: int = 0
        self._best_bic: float = np.inf
        self._training_date: Optional[datetime] = None
        self._feature_names: list[str] = []

        self._state_to_regime: dict[int, Regime] = {}
        self._state_to_label: dict[int, RegimeLabel] = {}
        self._regime_info: dict[int, RegimeInfo] = {}

        # Live-trading state tracking (updated by predict_last)
        self._state_history: list[int] = []
        self._confirmed_state: int = -1
        self._consecutive_count: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "HMMEngine":
        """
        Select the optimal HMM via BIC and train it on `features`.

        Tries each candidate in n_candidates.  For each candidate, runs
        n_init random restarts and keeps the best log-likelihood.  Selects
        the candidate with the lowest BIC (simplest model that explains data).

        All candidate BIC scores are logged at INFO level.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_bars, n_features).  Must have ≥ min_train_bars valid rows.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If features has fewer rows than min_train_bars.
        RuntimeError
            If no candidate model converges.
        """
        if len(features) < self.min_train_bars:
            raise ValueError(
                f"HMMEngine.fit requires ≥ {self.min_train_bars} bars, "
                f"got {len(features)}."
            )

        X = features.values.astype(float)
        n_samples, n_features = X.shape

        logger.info(
            "HMMEngine.fit: n_candidates=%s | n_samples=%d | n_features=%d",
            self.n_candidates,
            n_samples,
            n_features,
        )

        best_bic = np.inf
        best_model: Optional[GaussianHMM] = None
        best_n = self.n_candidates[0]

        for n in self.n_candidates:
            model, ll = self._fit_model(X, n)
            if model is None:
                logger.warning(
                    "  n_states=%d | all %d restarts failed — skipped",
                    n,
                    self.n_init,
                )
                continue
            bic = self._compute_bic(ll, n, n_features, n_samples)
            marker = "  ← best" if bic < best_bic else ""
            logger.info(
                "  n_states=%d | log_likelihood=%.4f | BIC=%.2f%s",
                n,
                ll,
                bic,
                marker,
            )
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n

        if best_model is None:
            raise RuntimeError(
                "HMMEngine.fit: no model converged for any candidate in n_candidates. "
                "Try reducing min_train_bars or increasing n_init."
            )

        self._model = best_model
        self._n_states = best_n
        self._best_bic = best_bic
        self._training_date = datetime.utcnow()
        self._feature_names = list(features.columns)

        self._state_to_label = self._label_states_by_return(best_model)
        self._state_to_regime = self._label_states_by_volatility(best_model)
        self._regime_info = self._build_regime_info(best_model)

        logger.info(
            "HMMEngine.fit complete: n_states=%d | BIC=%.2f | return_labels=%s | vol_labels=%s",
            best_n,
            best_bic,
            {k: v.value for k, v in self._state_to_label.items()},
            {k: v.value for k, v in self._state_to_regime.items()},
        )
        return self

    # ─────────────────────────────────────────────────────────────────
    # Inference — Forward Algorithm Only  (NO Viterbi / predict())
    # ─────────────────────────────────────────────────────────────────

    def predict_regime_filtered(self, features: pd.DataFrame) -> list[RegimeState]:
        """
        Compute P(state_t | obs_{0:t}) via the HMM forward algorithm.

        CRITICAL: this is the only inference method permitted in backtests and
        live trading.  It is strictly causal — state at t depends only on
        observations 0 … t.  Adding future observations never changes past
        posteriors.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (T, n_features).  DatetimeIndex expected.

        Returns
        -------
        list[RegimeState]
            One RegimeState per bar in chronological order.
        """
        self._check_fitted()
        X = features.values.astype(float)
        timestamps = (
            features.index if isinstance(features.index, pd.DatetimeIndex) else None
        )

        # Forward pass: (T, n_states) log-posteriors
        log_post = self._forward_pass(X)
        posteriors = np.exp(log_post)  # (T, n_states)

        # Raw most-probable state at each bar
        raw_states = np.argmax(posteriors, axis=1)  # (T,)

        # Stability filter: delay regime changes until they persist
        filtered, consec_arr, confirmed_arr = self._stability_filter_sequence(
            raw_states, self.stability_bars
        )

        results: list[RegimeState] = []
        for t in range(len(X)):
            sid = int(filtered[t])
            probs = posteriors[t]
            conf = float(probs[sid])

            if conf < self.min_confidence:
                label = RegimeLabel.UNKNOWN
                regime = Regime.UNKNOWN
            else:
                label = self._state_to_label.get(sid, RegimeLabel.UNKNOWN)
                regime = self._state_to_regime.get(sid, Regime.UNKNOWN)

            ts = timestamps[t] if timestamps is not None else None

            results.append(
                RegimeState(
                    label=label,
                    regime=regime,
                    state_id=sid,
                    probability=conf,
                    state_probabilities=probs.copy(),
                    timestamp=ts,
                    is_confirmed=bool(confirmed_arr[t]),
                    consecutive_bars=int(consec_arr[t]),
                )
            )

        return results

    def predict(self, features: pd.DataFrame) -> list[RegimeResult]:
        """
        Causal prediction for all bars (uses forward algorithm, not Viterbi).

        Parameters
        ----------
        features : pd.DataFrame

        Returns
        -------
        list[RegimeResult]
            Backward-compatible wrappers around the internal RegimeState list.
        """
        states = self.predict_regime_filtered(features)
        return [RegimeResult.from_regime_state(rs, self._n_states) for rs in states]

    def predict_last(self, features: pd.DataFrame) -> RegimeResult:
        """
        Return the current-bar regime and update live state tracking.

        Call this once per bar in the live trading loop.  Internally it runs
        the full forward pass over the provided history but returns only the
        final bar's result.  This is correct and necessary: the forward
        algorithm needs the full sequence to propagate alpha values.

        Parameters
        ----------
        features : pd.DataFrame
            Full history up to and including the current bar.

        Returns
        -------
        RegimeResult
        """
        states = self.predict_regime_filtered(features)
        last = states[-1]

        # Update live state history
        self._state_history.append(last.state_id)
        if len(self._state_history) > self.flicker_window:
            self._state_history.pop(0)

        # Log regime changes
        if last.is_confirmed and last.state_id != self._confirmed_state:
            if self._confirmed_state >= 0:
                old_label = self._state_to_label.get(
                    self._confirmed_state, RegimeLabel.UNKNOWN
                )
                logger.warning(
                    "Regime change CONFIRMED: %s → %s (state %d → %d)",
                    old_label.value,
                    last.label.value,
                    self._confirmed_state,
                    last.state_id,
                )
            else:
                logger.info(
                    "Regime initialised: %s (state %d, p=%.3f)",
                    last.label.value,
                    last.state_id,
                    last.probability,
                )
            self._confirmed_state = last.state_id
            self._consecutive_count = last.consecutive_bars
        elif last.state_id == self._confirmed_state:
            self._consecutive_count = last.consecutive_bars
            logger.info(
                "Regime stable: %s (consecutive=%d, p=%.3f)",
                last.label.value,
                last.consecutive_bars,
                last.probability,
            )

        return RegimeResult.from_regime_state(last, self._n_states)

    def predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return the full posterior probability matrix for all bars.

        Parameters
        ----------
        features : pd.DataFrame

        Returns
        -------
        np.ndarray
            Shape (T, n_states).  Row t = P(state | obs_{0:t}).
        """
        self._check_fitted()
        return np.exp(self._forward_pass(features.values.astype(float)))

    # ─────────────────────────────────────────────────────────────────
    # Live regime status queries
    # ─────────────────────────────────────────────────────────────────

    def get_regime_stability(self) -> int:
        """Consecutive bars spent in the current confirmed regime."""
        return self._consecutive_count

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Learned HMM transition probability matrix, shape (n_states, n_states)."""
        return self._model.transmat_.copy() if self._model is not None else None

    def detect_regime_change(self) -> bool:
        """
        Return True if the last stability_bars entries of _state_history all match
        and that state differs from the previously confirmed state.
        """
        if len(self._state_history) < self.stability_bars:
            return False
        recent = self._state_history[-self.stability_bars:]
        return len(set(recent)) == 1 and recent[-1] != self._confirmed_state

    def get_regime_flicker_rate(self) -> int:
        """Number of state transitions in the last flicker_window bars."""
        history = self._state_history[-self.flicker_window:]
        if len(history) < 2:
            return 0
        return sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])

    def is_flickering(self) -> bool:
        """Return True if the flicker rate exceeds flicker_threshold."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    def get_regime_info(self, state_id: int) -> Optional[RegimeInfo]:
        """Return the RegimeInfo for a given HMM state index."""
        return self._regime_info.get(state_id)

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Persist the fitted engine (model + metadata) to a pickle file.

        Parameters
        ----------
        path : str
            Destination file path, e.g. 'models/hmm_engine.pkl'.
        """
        self._check_fitted()
        payload = {
            "model": self._model,
            "n_states": self._n_states,
            "best_bic": self._best_bic,
            "training_date": self._training_date,
            "feature_names": self._feature_names,
            "state_to_regime": self._state_to_regime,
            "state_to_label": self._state_to_label,
            "regime_info": self._regime_info,
            # Hyperparameters (for reproducibility)
            "n_candidates": self.n_candidates,
            "n_init": self.n_init,
            "covariance_type": self.covariance_type,
            "min_train_bars": self.min_train_bars,
            "stability_bars": self.stability_bars,
            "flicker_window": self.flicker_window,
            "flicker_threshold": self.flicker_threshold,
            "min_confidence": self.min_confidence,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "HMMEngine saved: path=%s  n_states=%d  BIC=%.2f  trained=%s",
            path,
            self._n_states,
            self._best_bic,
            self._training_date,
        )

    @classmethod
    def load(cls, path: str) -> "HMMEngine":
        """
        Load a previously saved HMMEngine from a pickle file.

        Parameters
        ----------
        path : str

        Returns
        -------
        HMMEngine
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        engine = cls(
            n_candidates=payload.get("n_candidates", [3, 4, 5, 6, 7]),
            n_init=payload.get("n_init", 10),
            covariance_type=payload.get("covariance_type", "full"),
            min_train_bars=payload.get("min_train_bars", 504),
            stability_bars=payload.get("stability_bars", 3),
            flicker_window=payload.get("flicker_window", 20),
            flicker_threshold=payload.get("flicker_threshold", 4),
            min_confidence=payload.get("min_confidence", 0.55),
        )
        engine._model = payload["model"]
        engine._n_states = payload["n_states"]
        engine._best_bic = payload.get("best_bic", np.inf)
        engine._training_date = payload.get("training_date")
        engine._feature_names = payload.get("feature_names", [])
        engine._state_to_regime = payload["state_to_regime"]
        engine._state_to_label = payload["state_to_label"]
        engine._regime_info = payload.get("regime_info", {})

        logger.info(
            "HMMEngine loaded: path=%s  n_states=%d  trained=%s",
            path,
            engine._n_states,
            engine._training_date,
        )
        return engine

    # ─────────────────────────────────────────────────────────────────
    # Core inference — Forward Algorithm
    # ─────────────────────────────────────────────────────────────────

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Run the HMM forward (filtering) algorithm on observation sequence X.

        Computes the normalized log-posterior P(state_t | obs_{0:t}) at every t.

        Algorithm (log-space for numerical stability):
            α_0[j] = log π_j + log B_j(obs_0)
            α_t[j] = log B_j(obs_t) + logsumexp_i( α_{t-1}[i] + log A[i,j] )
            Normalize α_t so logsumexp(α_t) = 0 (i.e. probabilities sum to 1)

        where π = start probabilities, A = transition matrix, B = emission.

        Vectorized inner step avoids Python loops over states.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T, n_states)
            Normalized log-posteriors.  np.exp(result[t]) sums to 1 at each t.
        """
        T = X.shape[0]
        n_states = self._model.n_components

        # Log emission log P(obs_t | state_k) for all t, k  →  (T, n_states)
        log_emit = self._log_emission_probs(X)

        log_A = np.log(self._model.transmat_ + 1e-300)    # (n_states, n_states)
        log_pi = np.log(self._model.startprob_ + 1e-300)  # (n_states,)

        log_alpha = np.empty((T, n_states), dtype=np.float64)

        # ── t = 0 ─────────────────────────────────────────────────────
        log_alpha[0] = log_pi + log_emit[0]
        log_alpha[0] -= logsumexp(log_alpha[0])

        # ── t = 1 .. T−1 ──────────────────────────────────────────────
        # Vectorized forward step:
        #   log_alpha[t-1, :, None]  shape (n_states, 1)
        #   log_A                    shape (n_states, n_states)   A[from, to]
        #   broadcast sum            shape (n_states, n_states)
        #   logsumexp(axis=0)        shape (n_states,)  = sum over "from" states
        for t in range(1, T):
            log_alpha[t] = log_emit[t] + logsumexp(
                log_alpha[t - 1, :, np.newaxis] + log_A,
                axis=0,
            )
            log_alpha[t] -= logsumexp(log_alpha[t])

        return log_alpha

    def _log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(obs | state) for all (observation, state) pairs.

        Tries hmmlearn's internal vectorized method first (efficient Cholesky
        decomposition); falls back to scipy multivariate_normal if unavailable.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T, n_states)
        """
        try:
            return self._model._compute_log_likelihood(X)
        except AttributeError:
            pass

        # Fallback: iterate over states using scipy
        from scipy.stats import multivariate_normal  # noqa: PLC0415

        n_states = self._model.n_components
        T = X.shape[0]
        log_probs = np.full((T, n_states), -1e300, dtype=np.float64)
        for k in range(n_states):
            try:
                log_probs[:, k] = multivariate_normal.logpdf(
                    X,
                    mean=self._model.means_[k],
                    cov=self._model.covars_[k],
                    allow_singular=True,
                )
            except Exception as exc:
                logger.debug("_log_emission_probs state %d fallback failed: %s", k, exc)
        return log_probs

    # ─────────────────────────────────────────────────────────────────
    # Model selection & fitting
    # ─────────────────────────────────────────────────────────────────

    def _fit_model(
        self, X: np.ndarray, n_states: int
    ) -> tuple[Optional[GaussianHMM], float]:
        """
        Fit GaussianHMM with n_states across n_init random restarts.

        Returns the model with the highest total log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_states : int

        Returns
        -------
        tuple[GaussianHMM | None, float]
            (best_model, best_total_log_likelihood).
            best_model is None if all restarts failed.
        """
        best_model: Optional[GaussianHMM] = None
        best_ll = -np.inf

        for seed in range(self.n_init):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    tol=1e-4,
                    random_state=seed,
                    init_params="stmc",
                    params="stmc",
                )
                model.fit(X)
                # score() returns per-sample average LL; multiply for total
                total_ll = float(model.score(X)) * len(X)
                if total_ll > best_ll:
                    best_ll = total_ll
                    best_model = model
            except Exception as exc:
                logger.debug(
                    "_fit_model n_states=%d seed=%d failed: %s", n_states, seed, exc
                )

        return best_model, best_ll

    def _compute_bic(
        self,
        log_likelihood: float,
        n_states: int,
        n_features: int,
        n_samples: int,
    ) -> float:
        """
        BIC = −2 × log_likelihood + n_params × log(n_samples).

        Parameters
        ----------
        log_likelihood : float
            Total (not per-sample) log-likelihood.
        n_states : int
        n_features : int
        n_samples : int

        Returns
        -------
        float
        """
        n_params = self._n_params(n_states, n_features)
        return -2.0 * log_likelihood + n_params * np.log(n_samples)

    def _n_params(self, n_states: int, n_features: int) -> int:
        """Count free parameters for BIC (depends on covariance_type)."""
        start = n_states - 1                    # start probs (simplex → n-1 free)
        trans = n_states * (n_states - 1)       # each row of transmat (n-1 free)
        means = n_states * n_features
        if self.covariance_type == "full":
            covars = n_states * (n_features * (n_features + 1) // 2)
        elif self.covariance_type == "diag":
            covars = n_states * n_features
        elif self.covariance_type == "spherical":
            covars = n_states
        elif self.covariance_type == "tied":
            covars = n_features * (n_features + 1) // 2
        else:
            covars = n_states * n_features
        return start + trans + means + covars

    # ─────────────────────────────────────────────────────────────────
    # State labelling
    # ─────────────────────────────────────────────────────────────────

    def _label_states_by_return(self, model: GaussianHMM) -> dict[int, RegimeLabel]:
        """
        Map each HMM state → RegimeLabel by ranking mean returns.

        Assumes feature column 0 in model.means_ is the 1-period log return.
        Rank 0 = lowest return (CRASH/BEAR), rank n-1 = highest (BULL/EUPHORIA).

        Parameters
        ----------
        model : GaussianHMM

        Returns
        -------
        dict[int, RegimeLabel]
        """
        n_states = model.n_components
        label_names = _LABEL_MAP.get(n_states)
        if label_names is None:
            logger.warning(
                "_label_states_by_return: no label map for n_states=%d → UNKNOWN",
                n_states,
            )
            return {k: RegimeLabel.UNKNOWN for k in range(n_states)}

        # Feature 0 = log_ret_1
        return_means = model.means_[:, 0]
        sorted_idx = np.argsort(return_means)  # ascending

        mapping: dict[int, RegimeLabel] = {}
        for rank, state_idx in enumerate(sorted_idx):
            mapping[int(state_idx)] = RegimeLabel(label_names[rank])

        logger.debug(
            "_label_states_by_return: %s",
            {k: f"{v.value}({model.means_[k, 0]:.5f})" for k, v in mapping.items()},
        )
        return mapping

    def _label_states_by_volatility(self, model: GaussianHMM) -> dict[int, Regime]:
        """
        Map each HMM state → Regime (LOW/MID/HIGH_VOL) by volatility rank.

        Volatility proxy = mean diagonal of each state's covariance matrix
        (average per-feature variance).  Lowest → LOW_VOL; highest → HIGH_VOL;
        all middle states → MID_VOL.

        NOTE: This mapping is INDEPENDENT of the return-based label mapping.
        A state labelled BULL could be LOW_VOL or MID_VOL depending on data.

        Parameters
        ----------
        model : GaussianHMM

        Returns
        -------
        dict[int, Regime]
        """
        n_states = model.n_components
        vols = self._state_volatilities(model)
        sorted_idx = np.argsort(vols)  # ascending

        mapping: dict[int, Regime] = {}
        for rank, state_idx in enumerate(sorted_idx):
            if rank == 0:
                mapping[int(state_idx)] = Regime.LOW_VOL
            elif rank == n_states - 1:
                mapping[int(state_idx)] = Regime.HIGH_VOL
            else:
                mapping[int(state_idx)] = Regime.MID_VOL

        logger.debug(
            "_label_states_by_volatility: %s",
            {k: f"{v.value}(σ²={vols[k]:.5f})" for k, v in mapping.items()},
        )
        return mapping

    def _state_volatilities(self, model: GaussianHMM) -> np.ndarray:
        """
        Compute a scalar volatility proxy for each HMM state.

        Uses mean diagonal (average variance) of the covariance matrix —
        covariance_type agnostic.

        Parameters
        ----------
        model : GaussianHMM

        Returns
        -------
        np.ndarray, shape (n_states,)
        """
        n_states = model.n_components
        vols = np.zeros(n_states)

        if self.covariance_type == "full":
            for k in range(n_states):
                vols[k] = float(np.mean(np.diag(model.covars_[k])))
        elif self.covariance_type == "diag":
            vols = model.covars_.mean(axis=1)
        elif self.covariance_type == "spherical":
            vols = model.covars_.copy()
        elif self.covariance_type == "tied":
            # All states share one covariance matrix
            shared = float(np.mean(np.diag(model.covars_)))
            vols[:] = shared
        else:
            vols = np.ones(n_states)

        return vols

    def _build_regime_info(self, model: GaussianHMM) -> dict[int, RegimeInfo]:
        """
        Construct a RegimeInfo for each HMM state.

        Parameters
        ----------
        model : GaussianHMM

        Returns
        -------
        dict[int, RegimeInfo]
        """
        vols = self._state_volatilities(model)
        n_states = model.n_components

        # Default strategy config per volatility regime
        _defaults: dict[Regime, dict] = {
            Regime.LOW_VOL: {
                "type": "full_invest",
                "max_leverage": 1.25,
                "max_pos_pct": 0.15,
                "min_conf": 0.55,
            },
            Regime.MID_VOL: {
                "type": "reduced",
                "max_leverage": 1.0,
                "max_pos_pct": 0.12,
                "min_conf": 0.60,
            },
            Regime.HIGH_VOL: {
                "type": "defensive",
                "max_leverage": 1.0,
                "max_pos_pct": 0.10,
                "min_conf": 0.65,
            },
            Regime.UNKNOWN: {
                "type": "defensive",
                "max_leverage": 1.0,
                "max_pos_pct": 0.10,
                "min_conf": 0.70,
            },
        }

        info: dict[int, RegimeInfo] = {}
        for k in range(n_states):
            regime = self._state_to_regime.get(k, Regime.UNKNOWN)
            label = self._state_to_label.get(k, RegimeLabel.UNKNOWN)
            defs = _defaults[regime]

            # Annualise: feature-0 mean × 252 ≈ annual return
            ann_return = float(model.means_[k, 0]) * 252
            # Annualise: sqrt(mean_diag_variance) × sqrt(252) ≈ annual vol
            ann_vol = float(np.sqrt(max(vols[k], 1e-12)) * np.sqrt(252))

            info[k] = RegimeInfo(
                regime_id=k,
                regime_name=label.value,
                expected_return=ann_return,
                expected_volatility=ann_vol,
                recommended_strategy_type=defs["type"],
                max_leverage_allowed=defs["max_leverage"],
                max_position_size_pct=defs["max_pos_pct"],
                min_confidence_to_act=defs["min_conf"],
            )

        return info

    # ─────────────────────────────────────────────────────────────────
    # Stability / flicker filtering
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _stability_filter_sequence(
        raw_states: np.ndarray,
        stability_bars: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply stability filter: delay adoption of a new state until it has
        appeared for `stability_bars` consecutive raw bars.

        Parameters
        ----------
        raw_states : np.ndarray, shape (T,)
        stability_bars : int

        Returns
        -------
        filtered_states : np.ndarray[int], shape (T,)
        consecutive_bars : np.ndarray[int], shape (T,)
            How many consecutive bars `raw_states` has matched `filtered_states[t]`
            ending at t.
        is_confirmed : np.ndarray[bool], shape (T,)
            True when consecutive_bars >= stability_bars.
        """
        T = len(raw_states)
        filtered = np.empty(T, dtype=np.int64)
        consec = np.ones(T, dtype=np.int64)
        confirmed = np.zeros(T, dtype=bool)

        if T == 0:
            return filtered, consec, confirmed

        current = int(raw_states[0])
        filtered[0] = current
        consec[0] = 1
        confirmed[0] = stability_bars <= 1

        for t in range(1, T):
            s = int(raw_states[t])

            # Count consecutive occurrences of s ending at t (capped at stability_bars)
            run = 1
            for i in range(t - 1, max(t - stability_bars, -1), -1):
                if raw_states[i] == s:
                    run += 1
                else:
                    break

            # Adopt new state once it has persisted for stability_bars bars
            if run >= stability_bars and s != current:
                current = s

            filtered[t] = current

            # Count how many consecutive bars ending at t match `current`
            c = 0
            for i in range(t, -1, -1):
                if raw_states[i] == current:
                    c += 1
                else:
                    break

            consec[t] = c
            confirmed[t] = c >= stability_bars

        return filtered, consec, confirmed

    def _apply_stability_filter(
        self, states: np.ndarray, posteriors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Legacy helper: return (filtered_states, posteriors) unchanged.

        Full stability metadata is available via _stability_filter_sequence.
        """
        filtered, _, _ = self._stability_filter_sequence(states, self.stability_bars)
        return filtered, posteriors

    def _is_flickering(self) -> bool:
        """Proxy for is_flickering() used by legacy callers."""
        return self.is_flickering()

    # ─────────────────────────────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the engine has not been fitted."""
        if self._model is None:
            raise RuntimeError(
                "HMMEngine is not fitted.  Call fit() before predicting."
            )

    # ─────────────────────────────────────────────────────────────────
    # Legacy wrappers kept for skeleton back-compat
    # ─────────────────────────────────────────────────────────────────

    def _select_best_n_states(self, X: np.ndarray) -> int:
        """Legacy — called internally by fit() via _fit_model loop."""
        n_features = X.shape[1]
        n_samples = X.shape[0]
        best_bic = np.inf
        best_n = self.n_candidates[0]
        for n in self.n_candidates:
            model, ll = self._fit_model(X, n)
            if model is None:
                continue
            bic = self._compute_bic(ll, n, n_features, n_samples)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        return best_n
