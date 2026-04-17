"""
regime_strategies.py — Volatility-based allocation strategies.

Responsibilities:
  - Map a RegimeResult to a target portfolio allocation fraction and leverage.
  - Apply trend-confirmation logic for MID_VOL regimes.
  - Scale position sizes by confidence when regime is uncertain.
  - Expose a single entry point: get_allocation(regime_result, trend_signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.hmm_engine import Regime, RegimeResult


@dataclass
class AllocationDecision:
    """Output of the strategy layer for a single bar."""

    regime: Regime
    allocation_fraction: float   # Fraction of equity to deploy (0.0 – 1.0+)
    leverage: float              # Leverage multiplier to apply
    confidence_scaled: bool      # True if size was reduced due to low confidence
    notes: str = ""              # Human-readable reason for this decision


class RegimeStrategy:
    """
    Translates regime signals into target allocation fractions and leverage.

    Parameters
    ----------
    low_vol_allocation : float
        Allocation fraction in low-volatility / bull regime.
    mid_vol_allocation_trend : float
        Allocation fraction in mid-vol regime with confirmed uptrend.
    mid_vol_allocation_no_trend : float
        Allocation fraction in mid-vol regime without a clear trend.
    high_vol_allocation : float
        Allocation fraction in high-volatility / bear regime.
    low_vol_leverage : float
        Leverage multiplier applied in low-vol regime only.
    rebalance_threshold : float
        Minimum drift from target before triggering a rebalance.
    uncertainty_size_mult : float
        Multiplier applied to allocation when regime confidence < min_confidence.
    min_confidence : float
        Confidence threshold below which uncertainty scaling kicks in.
    """

    def __init__(
        self,
        low_vol_allocation: float = 0.95,
        mid_vol_allocation_trend: float = 0.95,
        mid_vol_allocation_no_trend: float = 0.60,
        high_vol_allocation: float = 0.60,
        low_vol_leverage: float = 1.25,
        rebalance_threshold: float = 0.10,
        uncertainty_size_mult: float = 0.50,
        min_confidence: float = 0.55,
    ) -> None:
        self.low_vol_allocation = low_vol_allocation
        self.mid_vol_allocation_trend = mid_vol_allocation_trend
        self.mid_vol_allocation_no_trend = mid_vol_allocation_no_trend
        self.high_vol_allocation = high_vol_allocation
        self.low_vol_leverage = low_vol_leverage
        self.rebalance_threshold = rebalance_threshold
        self.uncertainty_size_mult = uncertainty_size_mult
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_allocation(
        self,
        regime_result: RegimeResult,
        trend_signal: Optional[bool] = None,
    ) -> AllocationDecision:
        """
        Compute the target allocation fraction and leverage for one bar.

        Parameters
        ----------
        regime_result : RegimeResult
            Current regime detection output including confidence.
        trend_signal : bool | None
            True = uptrend confirmed, False = no trend / downtrend,
            None = unavailable (treated as no trend for mid-vol).

        Returns
        -------
        AllocationDecision
        """
        raise NotImplementedError

    def needs_rebalance(
        self,
        current_weight: float,
        target_weight: float,
    ) -> bool:
        """
        Return True if drift between current and target exceeds rebalance_threshold.

        Parameters
        ----------
        current_weight : float
            Current portfolio allocation fraction.
        target_weight : float
            Target allocation fraction from get_allocation().

        Returns
        -------
        bool
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_allocation(
        self, regime: Regime, trend_signal: Optional[bool]
    ) -> tuple[float, float]:
        """
        Return (allocation_fraction, leverage) for the given regime and trend.

        Parameters
        ----------
        regime : Regime
        trend_signal : bool | None

        Returns
        -------
        tuple[float, float]
            (allocation_fraction, leverage)
        """
        raise NotImplementedError

    def _apply_confidence_scaling(
        self,
        allocation: float,
        confidence: float,
    ) -> tuple[float, bool]:
        """
        Scale down allocation when confidence is below min_confidence.

        Parameters
        ----------
        allocation : float
            Raw allocation fraction.
        confidence : float
            Regime posterior probability.

        Returns
        -------
        tuple[float, bool]
            (scaled_allocation, was_scaled)
        """
        raise NotImplementedError
