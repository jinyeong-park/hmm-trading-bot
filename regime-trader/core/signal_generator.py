"""
signal_generator.py — Combines HMM regime detection and strategy into actionable signals.

Responsibilities:
  - Orchestrate the pipeline: features → HMM → strategy → risk check → signal.
  - Compute trend confirmation indicator (e.g., SMA slope or momentum filter).
  - Emit TradingSignal objects consumed by order_executor.
  - Maintain a per-symbol signal cache to avoid redundant reprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeResult
from core.regime_strategies import AllocationDecision, RegimeStrategy
from core.risk_manager import RiskCheckResult, RiskManager


class SignalAction(str, Enum):
    """High-level action directive passed to the order executor."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"       # Close all positions (risk halt)
    REBALANCE = "rebalance"


@dataclass
class TradingSignal:
    """Fully resolved signal ready for execution."""

    symbol: str
    action: SignalAction
    target_weight: float              # Desired portfolio weight (0.0–1.0+)
    regime_result: RegimeResult
    allocation_decision: AllocationDecision
    risk_check: RiskCheckResult
    trend_confirmed: Optional[bool]
    timestamp: pd.Timestamp
    notes: str = ""


class SignalGenerator:
    """
    Orchestrates HMM regime detection, strategy allocation, and risk gating
    into a single TradingSignal per symbol per bar.

    Parameters
    ----------
    hmm_engine : HMMEngine
        Fitted (or to-be-fitted) HMM regime detector.
    strategy : RegimeStrategy
        Allocation strategy that maps regimes to position sizes.
    risk_manager : RiskManager
        Risk gate that validates signals before they become orders.
    trend_lookback : int
        Number of bars used to compute the trend confirmation signal.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
        trend_lookback: int = 20,
    ) -> None:
        self.hmm_engine = hmm_engine
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.trend_lookback = trend_lookback

        self._signal_cache: dict[str, TradingSignal] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_positions: dict[str, float],
        equity: float,
    ) -> TradingSignal:
        """
        Run the full pipeline and return a TradingSignal for `symbol`.

        Parameters
        ----------
        symbol : str
            Ticker to generate a signal for.
        features : pd.DataFrame
            Feature matrix for `symbol` (output of FeatureEngineer).
        current_positions : dict[str, float]
            Map of symbol → current notional for all open positions.
        equity : float
            Current portfolio equity.

        Returns
        -------
        TradingSignal
        """
        raise NotImplementedError

    def generate_all(
        self,
        features_by_symbol: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
        equity: float,
    ) -> dict[str, TradingSignal]:
        """
        Generate signals for all symbols in `features_by_symbol`.

        Parameters
        ----------
        features_by_symbol : dict[str, pd.DataFrame]
        current_positions : dict[str, float]
        equity : float

        Returns
        -------
        dict[str, TradingSignal]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_trend_signal(self, features: pd.DataFrame) -> Optional[bool]:
        """
        Derive a boolean trend-confirmation signal from recent price data.

        Uses a simple SMA slope or momentum filter over trend_lookback bars.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain a 'close' column at minimum.

        Returns
        -------
        bool | None
            True = uptrend, False = no trend / downtrend, None = insufficient data.
        """
        raise NotImplementedError

    def _resolve_action(
        self,
        allocation: AllocationDecision,
        risk_check: RiskCheckResult,
        current_weight: float,
        target_weight: float,
    ) -> SignalAction:
        """
        Combine allocation decision and risk check into a final SignalAction.

        Parameters
        ----------
        allocation : AllocationDecision
        risk_check : RiskCheckResult
        current_weight : float
        target_weight : float

        Returns
        -------
        SignalAction
        """
        raise NotImplementedError
