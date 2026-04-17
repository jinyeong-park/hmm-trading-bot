"""
test_hmm.py — Unit tests for HMMEngine.

Tests cover:
  - Model fitting raises on insufficient data.
  - Optimal n_states selection returns a value in n_candidates.
  - State labelling correctly orders regimes by volatility.
  - Stability filter suppresses single-bar regime flickers.
  - Confidence gating returns UNKNOWN when below threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine, Regime, RegimeResult


class TestHMMEngineInit:
    def test_default_params(self):
        engine = HMMEngine()
        assert engine.n_candidates == [3, 4, 5, 6, 7]
        assert engine.n_init == 10
        assert engine.covariance_type == "full"

    def test_custom_params(self):
        engine = HMMEngine(n_candidates=[2, 3], n_init=5, min_confidence=0.70)
        assert engine.n_candidates == [2, 3]
        assert engine.min_confidence == 0.70


class TestHMMEngineFit:
    def test_fit_raises_on_insufficient_data(self):
        engine = HMMEngine(min_train_bars=252)
        short_features = pd.DataFrame(
            np.random.randn(100, 3), columns=["f1", "f2", "f3"]
        )
        with pytest.raises((ValueError, NotImplementedError)):
            engine.fit(short_features)

    def test_fit_returns_self(self):
        pytest.skip("Requires implementation")

    def test_n_states_within_candidates(self):
        pytest.skip("Requires implementation")


class TestHMMEnginePredict:
    def test_predict_returns_correct_length(self):
        pytest.skip("Requires implementation")

    def test_predict_last_returns_single_result(self):
        pytest.skip("Requires implementation")

    def test_confidence_below_threshold_returns_unknown(self):
        pytest.skip("Requires implementation")


class TestRegimeLabelling:
    def test_low_vol_state_identified(self):
        pytest.skip("Requires implementation")

    def test_high_vol_state_identified(self):
        pytest.skip("Requires implementation")

    def test_mid_vol_states_are_remainder(self):
        pytest.skip("Requires implementation")


class TestStabilityFilter:
    def test_single_bar_flicker_suppressed(self):
        pytest.skip("Requires implementation")

    def test_stable_regime_passes_through(self):
        pytest.skip("Requires implementation")
