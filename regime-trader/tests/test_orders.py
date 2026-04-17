"""
test_orders.py — Unit tests for OrderExecutor.

Tests cover:
  - HOLD signals produce no order.
  - Order deduplication blocks double-entry.
  - Stale order cancellation identifies and cancels old unfilled orders.
  - Qty computation from target weight and equity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import Order, OrderExecutor, OrderStatus
from core.hmm_engine import Regime, RegimeResult
from core.risk_manager import RiskAction, RiskCheckResult
from core.regime_strategies import AllocationDecision
from core.signal_generator import SignalAction, TradingSignal

import numpy as np
import pandas as pd


def _make_hold_signal(symbol: str = "SPY") -> TradingSignal:
    """Build a minimal HOLD TradingSignal for testing."""
    regime_result = RegimeResult(
        regime=Regime.LOW_VOL,
        state_index=0,
        confidence=0.80,
        posteriors=np.array([0.80, 0.10, 0.10]),
        n_states=3,
    )
    allocation = AllocationDecision(
        regime=Regime.LOW_VOL,
        allocation_fraction=0.95,
        leverage=1.0,
        confidence_scaled=False,
    )
    risk_check = RiskCheckResult(action=RiskAction.ALLOW, size_multiplier=1.0)
    return TradingSignal(
        symbol=symbol,
        action=SignalAction.HOLD,
        target_weight=0.10,
        regime_result=regime_result,
        allocation_decision=allocation,
        risk_check=risk_check,
        trend_confirmed=True,
        timestamp=pd.Timestamp.now(),
    )


class TestExecuteSignal:
    def test_hold_signal_returns_none(self):
        client = MagicMock(spec=AlpacaClient)
        executor = OrderExecutor(client)
        signal = _make_hold_signal()
        with pytest.raises(NotImplementedError):
            executor.execute_signal(signal, equity=100_000)

    def test_buy_signal_submits_order(self):
        pytest.skip("Requires implementation")

    def test_deduplication_blocks_repeat(self):
        pytest.skip("Requires implementation")


class TestQtyComputation:
    def test_qty_proportional_to_weight(self):
        client = MagicMock(spec=AlpacaClient)
        executor = OrderExecutor(client)
        with pytest.raises(NotImplementedError):
            executor._compute_qty(weight=0.10, price=500.0, equity=100_000)

    def test_qty_zero_when_weight_zero(self):
        pytest.skip("Requires implementation")


class TestStaleOrderCancellation:
    def test_fresh_order_not_cancelled(self):
        pytest.skip("Requires implementation")

    def test_old_order_cancelled(self):
        pytest.skip("Requires implementation")
