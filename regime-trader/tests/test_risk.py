"""
test_risk.py — Unit tests for RiskManager.

Tests cover:
  - Position sizing from risk fraction and ATR stop.
  - Max single position, max exposure, and max concurrent limits.
  - Daily/weekly drawdown circuit breakers (reduce and halt).
  - Peak-to-trough drawdown halt.
  - Daily trade counter and max_daily_trades enforcement.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.risk_manager import RiskAction, RiskManager


class TestPositionSizing:
    def test_basic_size_calculation(self):
        rm = RiskManager(max_risk_per_trade=0.01)
        with pytest.raises(NotImplementedError):
            rm.compute_position_size(
                equity=100_000,
                entry_price=100.0,
                stop_price=98.0,
                allocation_fraction=0.95,
            )

    def test_size_zero_when_stop_equals_entry(self):
        pytest.skip("Requires implementation")

    def test_size_capped_by_allocation(self):
        pytest.skip("Requires implementation")


class TestExposureLimits:
    def test_rejects_when_over_max_exposure(self):
        pytest.skip("Requires implementation")

    def test_rejects_when_single_position_too_large(self):
        pytest.skip("Requires implementation")

    def test_rejects_when_max_concurrent_reached(self):
        pytest.skip("Requires implementation")


class TestDrawdownCircuitBreakers:
    def test_daily_reduce_threshold(self):
        rm = RiskManager(daily_dd_reduce=0.02, daily_dd_halt=0.03)
        ts = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        rm.update_equity(100_000, ts)
        with pytest.raises(NotImplementedError):
            rm.update_equity(97_500, ts.replace(hour=15))

    def test_daily_halt_threshold(self):
        pytest.skip("Requires implementation")

    def test_weekly_reduce_threshold(self):
        pytest.skip("Requires implementation")

    def test_peak_drawdown_halt(self):
        pytest.skip("Requires implementation")


class TestDailyTradeCounter:
    def test_counter_increments(self):
        rm = RiskManager(max_daily_trades=20)
        with pytest.raises(NotImplementedError):
            rm.register_trade()

    def test_rejects_when_max_trades_reached(self):
        pytest.skip("Requires implementation")

    def test_counter_resets_next_day(self):
        pytest.skip("Requires implementation")
