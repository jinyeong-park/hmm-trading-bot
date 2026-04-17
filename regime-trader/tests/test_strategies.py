"""
test_strategies.py — Unit tests for RegimeStrategy allocation logic.

Tests cover:
  - Correct allocation fraction and leverage per regime × trend combination.
  - Confidence scaling reduces allocation below min_confidence threshold.
  - Rebalance threshold logic triggers/suppresses rebalances correctly.
"""

from __future__ import annotations

import pytest

from core.hmm_engine import Regime, RegimeResult
from core.regime_strategies import AllocationDecision, RegimeStrategy

import numpy as np


def _make_regime_result(regime: Regime, confidence: float) -> RegimeResult:
    """Helper: build a minimal RegimeResult for strategy tests."""
    n_states = 3
    posteriors = np.zeros(n_states)
    posteriors[0] = confidence
    return RegimeResult(
        regime=regime,
        state_index=0,
        confidence=confidence,
        posteriors=posteriors,
        n_states=n_states,
    )


class TestAllocationByRegime:
    def test_low_vol_full_allocation(self):
        strategy = RegimeStrategy(low_vol_allocation=0.95, min_confidence=0.55)
        result = _make_regime_result(Regime.LOW_VOL, confidence=0.80)
        with pytest.raises(NotImplementedError):
            strategy.get_allocation(result, trend_signal=True)

    def test_mid_vol_with_trend(self):
        pytest.skip("Requires implementation")

    def test_mid_vol_no_trend(self):
        pytest.skip("Requires implementation")

    def test_high_vol_defensive_allocation(self):
        pytest.skip("Requires implementation")

    def test_unknown_regime_uses_low_allocation(self):
        pytest.skip("Requires implementation")


class TestLeverage:
    def test_low_vol_leverage_applied(self):
        pytest.skip("Requires implementation")

    def test_high_vol_no_leverage(self):
        pytest.skip("Requires implementation")


class TestConfidenceScaling:
    def test_low_confidence_scales_down(self):
        strategy = RegimeStrategy(uncertainty_size_mult=0.50, min_confidence=0.55)
        result = _make_regime_result(Regime.LOW_VOL, confidence=0.50)
        with pytest.raises(NotImplementedError):
            strategy.get_allocation(result)

    def test_high_confidence_no_scaling(self):
        pytest.skip("Requires implementation")


class TestRebalanceThreshold:
    def test_within_threshold_no_rebalance(self):
        strategy = RegimeStrategy(rebalance_threshold=0.10)
        with pytest.raises(NotImplementedError):
            strategy.needs_rebalance(current_weight=0.90, target_weight=0.95)

    def test_beyond_threshold_triggers_rebalance(self):
        pytest.skip("Requires implementation")
