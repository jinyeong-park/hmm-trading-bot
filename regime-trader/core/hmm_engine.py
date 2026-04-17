"""
hmm_engine.py — HMM regime detection engine.

Responsibilities:
  - Fit a Gaussian HMM (or GaussianMixture HMM) on a feature matrix.
  - Select the optimal number of hidden states via BIC / AIC across n_candidates.
  - Decode the most likely state sequence (Viterbi).
  - Label states as low-vol / mid-vol / high-vol regimes by volatility rank.
  - Detect regime flickering and apply a stability filter.
  - Return posterior probabilities for downstream confidence gating.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class Regime(str, Enum):
    """Canonical regime labels ordered by expected volatility."""

    LOW_VOL = "low_vol"
    MID_VOL = "mid_vol"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


class RegimeResult:
    """Container for a single regime detection output."""

    def __init__(
        self,
        regime: Regime,
        state_index: int,
        confidence: float,
        posteriors: np.ndarray,
        n_states: int,
    ) -> None:
        self.regime = regime
        self.state_index = state_index
        self.confidence = confidence          # max posterior probability
        self.posteriors = posteriors          # full posterior vector (n_states,)
        self.n_states = n_states

    def __repr__(self) -> str:
        return (
            f"RegimeResult(regime={self.regime}, state={self.state_index}, "
            f"confidence={self.confidence:.3f})"
        )


class HMMEngine:
    """
    Fits a GaussianHMM and decodes market regimes from feature data.

    Parameters
    ----------
    n_candidates : list[int]
        Candidate state counts evaluated during model selection.
    n_init : int
        Number of random restarts per candidate (avoids local optima).
    covariance_type : str
        HMM covariance structure — 'full' | 'diag' | 'spherical' | 'tied'.
    min_train_bars : int
        Minimum rows of feature data required before fitting.
    stability_bars : int
        Consecutive bars a regime must persist before being confirmed.
    flicker_window : int
        Rolling window size for counting regime transitions.
    flicker_threshold : int
        Maximum allowed transitions in flicker_window before dampening.
    min_confidence : float
        Minimum posterior probability needed to emit a non-UNKNOWN signal.
    """

    def __init__(
        self,
        n_candidates: list[int] | None = None,
        n_init: int = 10,
        covariance_type: str = "full",
        min_train_bars: int = 252,
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

        self._model: Optional[GaussianHMM] = None
        self._n_states: int = 0
        self._state_to_regime: dict[int, Regime] = {}
        self._state_history: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> "HMMEngine":
        """
        Select the best-fitting HMM and train it on `features`.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_bars, n_features). Must have >= min_train_bars rows.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> list[RegimeResult]:
        """
        Decode regimes for every bar in `features`.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_bars, n_features).

        Returns
        -------
        list[RegimeResult]
            One RegimeResult per bar, in chronological order.
        """
        raise NotImplementedError

    def predict_last(self, features: pd.DataFrame) -> RegimeResult:
        """
        Return the RegimeResult for only the most recent bar.

        Parameters
        ----------
        features : pd.DataFrame
            Must include all history needed for a stable Viterbi decode.

        Returns
        -------
        RegimeResult
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _select_best_n_states(self, X: np.ndarray) -> int:
        """
        Try each candidate state count and return the one with lowest BIC.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features).

        Returns
        -------
        int
            Optimal number of hidden states.
        """
        raise NotImplementedError

    def _fit_model(self, X: np.ndarray, n_states: int) -> GaussianHMM:
        """
        Fit a GaussianHMM with `n_states` states using multiple random restarts.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features).
        n_states : int
            Number of hidden states.

        Returns
        -------
        GaussianHMM
            The best model found across n_init restarts.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State labelling
    # ------------------------------------------------------------------

    def _label_states_by_volatility(self, model: GaussianHMM) -> dict[int, Regime]:
        """
        Map each HMM state index to a Regime by ranking states on volatility.

        Low-volatility state  → Regime.LOW_VOL
        Middle state(s)       → Regime.MID_VOL
        High-volatility state → Regime.HIGH_VOL

        Parameters
        ----------
        model : GaussianHMM
            Fitted model whose means/covars encode per-state distributions.

        Returns
        -------
        dict[int, Regime]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Stability / flicker filtering
    # ------------------------------------------------------------------

    def _apply_stability_filter(
        self, states: np.ndarray, posteriors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Suppress short-lived state changes that don't persist for stability_bars.

        Parameters
        ----------
        states : np.ndarray
            Raw Viterbi state sequence, shape (n_bars,).
        posteriors : np.ndarray
            Posterior probabilities, shape (n_bars, n_states).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered (states, posteriors).
        """
        raise NotImplementedError

    def _is_flickering(self) -> bool:
        """
        Return True if recent state_history has too many transitions.

        Uses the last flicker_window entries of _state_history and counts
        transitions. Returns True if count exceeds flicker_threshold.
        """
        raise NotImplementedError
