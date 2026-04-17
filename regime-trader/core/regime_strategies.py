"""
regime_strategies.py — Volatility-based allocation strategies.

DESIGN PHILOSOPHY
─────────────────
The HMM detects VOLATILITY ENVIRONMENTS, not market direction.
Stocks trend upward ~70% of the time in low-volatility periods.
The worst drawdowns cluster in high-volatility spikes.

Strategy:
  Low vol  → fully invested (calm markets compound)
  Mid vol  → stay invested if trend intact, reduce if broken
  High vol → reduce but stay long (catch V-shaped rebounds)

ALWAYS LONG. NEVER SHORT.
Shorting was tested extensively in walk-forward backtesting and
consistently destroyed returns due to:
  1. Markets have long-term upward drift
  2. V-shaped recoveries happen fast; HMM is 2–3 days late
  3. Short positions during rebounds wipe out crash gains

The correct response to high volatility is REDUCING allocation, not reversing.

VOL RANK MAPPING (for any n_regimes)
─────────────────────────────────────
  position = rank / (n_regimes − 1)   # 0.0 = calmest, 1.0 = most volatile
  position ≤ 0.33  →  LowVolBullStrategy
  position ≥ 0.67  →  HighVolDefensiveStrategy
  else             →  MidVolCautiousStrategy

This sort is INDEPENDENT of the return-based RegimeLabel sort.
"BULL" label does NOT mean low vol — the orchestrator ignores labels.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd

from core.hmm_engine import Regime, RegimeInfo, RegimeLabel, RegimeResult, RegimeState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ─────────────────────────────────────────────────────────────────────────────

class SignalDirection(str, Enum):
    """Position direction emitted by the strategy layer."""

    LONG = "LONG"
    FLAT = "FLAT"   # Hold no position (extreme uncertainty / risk halt)


@dataclass
class Signal:
    """Fully-resolved trading signal ready for the order executor."""

    symbol: str
    direction: SignalDirection
    confidence: float               # HMM posterior probability
    entry_price: float              # Latest close (order reference price)
    stop_loss: Optional[float]      # Stop-loss price (None if bars unavailable)
    take_profit: Optional[float]    # Take-profit price (optional)
    position_size_pct: float        # Fraction of portfolio equity to allocate
    leverage: float                 # Leverage multiplier (1.0 = no leverage)
    regime_id: int                  # HMM state index
    regime_name: str                # RegimeLabel.value (human label)
    regime_probability: float       # HMM posterior for this state
    timestamp: Optional[pd.Timestamp]
    reasoning: str                  # Plain-English explanation of the decision
    strategy_name: str              # Which strategy class produced this signal
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible allocation dataclass (used by signal_generator)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllocationDecision:
    """Output of the legacy RegimeStrategy.get_allocation() API."""

    regime: Regime
    allocation_fraction: float   # Fraction of equity to deploy (0.0 – 1.0+)
    leverage: float              # Leverage multiplier
    confidence_scaled: bool      # True if size was reduced due to low confidence
    notes: str = ""              # Human-readable reason for this decision


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-layer constants
# ─────────────────────────────────────────────────────────────────────────────

_LOW_VOL_THRESHOLD: float = 0.33
_HIGH_VOL_THRESHOLD: float = 0.67

# Default allocations
_LOW_ALLOC: float = 0.95
_LOW_LEVERAGE: float = 1.25

_MID_ALLOC_TREND: float = 0.95
_MID_LEVERAGE_TREND: float = 1.6
_MID_ALLOC_NO_TREND: float = 0.60
_MID_LEVERAGE_NO_TREND: float = 1.0

_HIGH_ALLOC: float = 0.60
_HIGH_LEVERAGE: float = 1.0

_EMA_WINDOW: int = 50
_ATR_WINDOW: int = 14
_MIN_CONFIDENCE: float = 0.55
_UNCERTAINTY_MULT: float = 0.50
_REBALANCE_THRESHOLD: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Pure bar-computation helpers (no external library dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ema(close: pd.Series, window: int = 50) -> float:
    """Return the most recent value of the exponential moving average."""
    return float(close.astype(float).ewm(span=window, adjust=False).mean().iloc[-1])


def _compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> float:
    """
    Return the most recent ATR using Wilder smoothing (EMA of true range).

    True range = max(H−L, |H−prev_C|, |L−prev_C|)
    """
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(span=window, adjust=False).mean()
    return float(atr.iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base strategy
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base class for all regime-specific allocation strategies.

    Subclasses implement compute_allocation() and optionally compute_stops().
    generate_signal() is the single public method used by the orchestrator.
    """

    name: str = "BaseStrategy"

    def __init__(
        self,
        min_confidence: float = _MIN_CONFIDENCE,
        uncertainty_size_mult: float = _UNCERTAINTY_MULT,
        ema_window: int = _EMA_WINDOW,
        atr_window: int = _ATR_WINDOW,
    ) -> None:
        self.min_confidence = min_confidence
        self.uncertainty_size_mult = uncertainty_size_mult
        self.ema_window = ema_window
        self.atr_window = atr_window

    # ── Abstract methods ──────────────────────────────────────────────

    @abstractmethod
    def compute_allocation(
        self,
        bars: Optional[pd.DataFrame],
        regime_state: RegimeState,
    ) -> tuple[float, float, str]:
        """
        Return (position_size_pct, leverage, reasoning) for the current bar.

        Parameters
        ----------
        bars : pd.DataFrame | None
            OHLCV DataFrame.  None if unavailable.
        regime_state : RegimeState
            Current regime snapshot from HMMEngine.

        Returns
        -------
        tuple[float, float, str]
            (position_size_pct, leverage, reasoning)
        """

    # ── Optional override ─────────────────────────────────────────────

    def compute_stops(
        self,
        bars: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Return (stop_loss, take_profit) prices.

        Returns (None, None) if bars is unavailable or has insufficient history.
        Override in subclasses for regime-specific stop placement.
        """
        return None, None

    # ── Main entry point ──────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        bars: Optional[pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> Signal:
        """
        Produce a Signal for `symbol` given the current regime state.

        Applies uncertainty scaling automatically when confidence is below
        threshold or the regime is not yet confirmed or the HMM is flickering.

        Parameters
        ----------
        symbol : str
        bars : pd.DataFrame | None
            OHLCV data used for EMA/ATR stop computation and trend detection.
        regime_state : RegimeState
        is_flickering : bool
            True if HMMEngine.is_flickering() returns True.

        Returns
        -------
        Signal
        """
        size_pct, leverage, reasoning = self.compute_allocation(bars, regime_state)
        stop_loss, take_profit = self.compute_stops(bars)

        # ── Uncertainty mode ─────────────────────────────────────────
        uncertainty = (
            regime_state.probability < self.min_confidence
            or not regime_state.is_confirmed
            or is_flickering
        )
        if uncertainty:
            size_pct = round(size_pct * self.uncertainty_size_mult, 4)
            leverage = 1.0
            reasoning += " [UNCERTAINTY — size halved]"

        # ── Entry price & timestamp ──────────────────────────────────
        if bars is not None and len(bars) > 0:
            entry_price = float(bars["close"].iloc[-1])
            ts = bars.index[-1] if isinstance(bars.index, pd.DatetimeIndex) else None
        else:
            entry_price = 0.0
            ts = None

        return Signal(
            symbol=symbol,
            direction=SignalDirection.LONG,  # ALWAYS LONG
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=round(size_pct, 4),
            leverage=round(leverage, 4),
            regime_id=regime_state.state_id,
            regime_name=regime_state.label.value,
            regime_probability=regime_state.probability,
            timestamp=ts,
            reasoning=reasoning,
            strategy_name=self.name,
        )

    # ── Shared utility ────────────────────────────────────────────────

    def is_above_ema(self, bars: Optional[pd.DataFrame]) -> bool:
        """
        Return True if the latest close is above EMA(ema_window).

        Defaults to True (trend intact) when bars are unavailable or too short.
        """
        if bars is None or len(bars) < self.ema_window:
            return True
        return float(bars["close"].iloc[-1]) > _compute_ema(bars["close"], self.ema_window)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete strategy classes
# ─────────────────────────────────────────────────────────────────────────────

class LowVolBullStrategy(BaseStrategy):
    """
    Low-volatility bull regime strategy.

    Vol rank ≤ 0.33 — calm markets trend up.  This is where the bulk of
    returns are generated through full allocation and modest leverage.

    Allocation : 95%
    Leverage   : 1.25×
    Stop loss  : max(close − 3×ATR,  EMA50 − 0.5×ATR)
    """

    name = "LowVolBullStrategy"

    def compute_allocation(
        self,
        bars: Optional[pd.DataFrame],
        regime_state: RegimeState,
    ) -> tuple[float, float, str]:
        reasoning = (
            f"Low-vol regime (state={regime_state.state_id} "
            f"'{regime_state.label.value}', "
            f"p={regime_state.probability:.3f}): "
            f"full {_LOW_ALLOC:.0%} allocation × {_LOW_LEVERAGE}× leverage"
        )
        return _LOW_ALLOC, _LOW_LEVERAGE, reasoning

    def compute_stops(
        self,
        bars: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        if bars is None or len(bars) < max(self.ema_window, self.atr_window) + 5:
            return None, None
        try:
            close = float(bars["close"].iloc[-1])
            ema = _compute_ema(bars["close"], self.ema_window)
            atr = _compute_atr(bars, self.atr_window)
            # Wider of the two anchor points: momentum stop or EMA stop
            stop = max(close - 3.0 * atr, ema - 0.5 * atr)
            return round(stop, 6), None
        except Exception as exc:
            logger.debug("LowVolBullStrategy.compute_stops error: %s", exc)
            return None, None


class MidVolCautiousStrategy(BaseStrategy):
    """
    Mid-volatility cautious regime strategy.

    Vol rank in (0.33, 0.67) — conditionally invested depending on trend.

    Trend intact (close > EMA50) : 95% allocation, 1.6× leverage
    Trend broken (close ≤ EMA50) : 60% allocation, 1.0× leverage
    Stop loss : EMA50 − 0.5×ATR
    """

    name = "MidVolCautiousStrategy"

    def compute_allocation(
        self,
        bars: Optional[pd.DataFrame],
        regime_state: RegimeState,
    ) -> tuple[float, float, str]:
        trend_up = self.is_above_ema(bars)

        if trend_up:
            size_pct = _MID_ALLOC_TREND
            leverage = _MID_LEVERAGE_TREND
            trend_str = f"trend INTACT (close > EMA{self.ema_window})"
        else:
            size_pct = _MID_ALLOC_NO_TREND
            leverage = _MID_LEVERAGE_NO_TREND
            trend_str = f"trend BROKEN (close ≤ EMA{self.ema_window})"

        reasoning = (
            f"Mid-vol regime (state={regime_state.state_id} "
            f"'{regime_state.label.value}', "
            f"p={regime_state.probability:.3f}), {trend_str}: "
            f"{size_pct:.0%} × {leverage}× leverage"
        )
        return size_pct, leverage, reasoning

    def compute_stops(
        self,
        bars: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        if bars is None or len(bars) < max(self.ema_window, self.atr_window) + 5:
            return None, None
        try:
            ema = _compute_ema(bars["close"], self.ema_window)
            atr = _compute_atr(bars, self.atr_window)
            stop = ema - 0.5 * atr
            return round(stop, 6), None
        except Exception as exc:
            logger.debug("MidVolCautiousStrategy.compute_stops error: %s", exc)
            return None, None


class HighVolDefensiveStrategy(BaseStrategy):
    """
    High-volatility defensive regime strategy.

    Vol rank ≥ 0.67 — turbulent conditions.  ALWAYS LONG (never short).
    60% allocation stays invested to capture V-shaped rebounds while
    limiting exposure to continued drawdowns.

    Allocation : 60%
    Leverage   : 1.0× (no amplification during turbulence)
    Stop loss  : EMA50 − 1.0×ATR  (wider to absorb volatility spikes)
    """

    name = "HighVolDefensiveStrategy"

    def compute_allocation(
        self,
        bars: Optional[pd.DataFrame],
        regime_state: RegimeState,
    ) -> tuple[float, float, str]:
        reasoning = (
            f"High-vol regime (state={regime_state.state_id} "
            f"'{regime_state.label.value}', "
            f"p={regime_state.probability:.3f}): "
            f"defensive {_HIGH_ALLOC:.0%} allocation, "
            f"no leverage — LONG only, never short"
        )
        return _HIGH_ALLOC, _HIGH_LEVERAGE, reasoning

    def compute_stops(
        self,
        bars: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        if bars is None or len(bars) < max(self.ema_window, self.atr_window) + 5:
            return None, None
        try:
            ema = _compute_ema(bars["close"], self.ema_window)
            atr = _compute_atr(bars, self.atr_window)
            # Wider stop to avoid getting shaken out during spikes
            stop = ema - 1.0 * atr
            return round(stop, 6), None
        except Exception as exc:
            logger.debug("HighVolDefensiveStrategy.compute_stops error: %s", exc)
            return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Label-to-strategy fallback mapping
# Purpose: used when vol_rank mapping is unavailable (e.g. cold start)
# Note: the orchestrator uses vol rank, NOT this dict, as primary routing
# ─────────────────────────────────────────────────────────────────────────────

LABEL_TO_STRATEGY: dict[RegimeLabel, type[BaseStrategy]] = {
    RegimeLabel.CRASH:       HighVolDefensiveStrategy,
    RegimeLabel.STRONG_BEAR: HighVolDefensiveStrategy,
    RegimeLabel.WEAK_BEAR:   HighVolDefensiveStrategy,
    RegimeLabel.BEAR:        HighVolDefensiveStrategy,
    RegimeLabel.NEUTRAL:     MidVolCautiousStrategy,
    RegimeLabel.WEAK_BULL:   MidVolCautiousStrategy,
    RegimeLabel.BULL:        LowVolBullStrategy,
    RegimeLabel.STRONG_BULL: LowVolBullStrategy,
    RegimeLabel.EUPHORIA:    LowVolBullStrategy,
    RegimeLabel.UNKNOWN:     HighVolDefensiveStrategy,   # fail-safe: defensive
}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible name aliases
# ─────────────────────────────────────────────────────────────────────────────

CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Strategy orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class StrategyOrchestrator:
    """
    Maps each HMM regime_id to a strategy instance via volatility rank.

    Sorting regime_infos by expected_volatility (ascending) produces a
    vol_rank in [0.0, 1.0] for each regime.  This sort is INDEPENDENT of
    the return-based RegimeLabel order — the orchestrator ignores labels.

    Parameters
    ----------
    config : dict | None
        Optional settings.yaml-derived config dict.  Reads from
        config['strategy'] for allocation defaults.
    regime_infos : dict[int, RegimeInfo] | None
        Mapping regime_id → RegimeInfo from the fitted HMM.  Can be
        provided later via update_regime_infos().
    min_confidence : float
    uncertainty_size_mult : float
    rebalance_threshold : float
    ema_window : int
    atr_window : int
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        regime_infos: Optional[dict[int, RegimeInfo]] = None,
        min_confidence: float = _MIN_CONFIDENCE,
        uncertainty_size_mult: float = _UNCERTAINTY_MULT,
        rebalance_threshold: float = _REBALANCE_THRESHOLD,
        ema_window: int = _EMA_WINDOW,
        atr_window: int = _ATR_WINDOW,
    ) -> None:
        cfg = (config or {}).get("strategy", {})

        self.min_confidence = cfg.get("min_confidence", min_confidence)
        self.uncertainty_size_mult = cfg.get(
            "uncertainty_size_mult", uncertainty_size_mult
        )
        self.rebalance_threshold = cfg.get("rebalance_threshold", rebalance_threshold)
        self.ema_window = ema_window
        self.atr_window = atr_window

        self._regime_to_strategy: dict[int, BaseStrategy] = {}
        self._vol_ranks: dict[int, float] = {}

        if regime_infos:
            self.update_regime_infos(regime_infos)

    # ── Core API ──────────────────────────────────────────────────────

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """
        Rebuild the regime_id → strategy mapping after HMM retraining.

        Parameters
        ----------
        regime_infos : dict[int, RegimeInfo]
            Fresh mapping from the re-fitted HMM engine.
        """
        if not regime_infos:
            logger.warning("StrategyOrchestrator.update_regime_infos: empty input")
            return

        n = len(regime_infos)
        # Sort by expected_volatility ascending (independent of label/return sort)
        sorted_ids: list[int] = sorted(
            regime_infos.keys(),
            key=lambda rid: regime_infos[rid].expected_volatility,
        )

        self._vol_ranks = {}
        self._regime_to_strategy = {}

        for rank, regime_id in enumerate(sorted_ids):
            position = rank / max(n - 1, 1)
            self._vol_ranks[regime_id] = position
            strategy = self._make_strategy(position)
            self._regime_to_strategy[regime_id] = strategy

            logger.debug(
                "  regime_id=%-2d  name=%-14s  vol=%.4f  vol_rank=%.3f  → %s",
                regime_id,
                regime_infos[regime_id].regime_name,
                regime_infos[regime_id].expected_volatility,
                position,
                strategy.name,
            )

        logger.info(
            "StrategyOrchestrator updated: %d regimes → %s",
            n,
            {rid: self._regime_to_strategy[rid].name for rid in sorted_ids},
        )

    def generate_signals(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> list[Signal]:
        """
        Generate a Signal for each symbol using the regime-appropriate strategy.

        Parameters
        ----------
        symbols : list[str]
        bars_by_symbol : dict[str, pd.DataFrame]
            Mapping symbol → OHLCV DataFrame.
        regime_state : RegimeState
            Current regime from HMMEngine.predict_last().
        is_flickering : bool
            Pass HMMEngine.is_flickering() result here.

        Returns
        -------
        list[Signal]
        """
        strategy = self._get_strategy(regime_state.state_id, regime_state.label)
        signals: list[Signal] = []

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            try:
                sig = strategy.generate_signal(
                    symbol=symbol,
                    bars=bars,
                    regime_state=regime_state,
                    is_flickering=is_flickering,
                )
                signals.append(sig)
                logger.debug(
                    "Signal: %s  %s  size=%.1f%%  lev=%.2f×  stop=%s  reason: %s",
                    sig.symbol,
                    sig.direction.value,
                    sig.position_size_pct * 100,
                    sig.leverage,
                    f"{sig.stop_loss:.2f}" if sig.stop_loss else "N/A",
                    sig.reasoning,
                )
            except Exception as exc:
                logger.error(
                    "StrategyOrchestrator: signal failed for %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )

        return signals

    def needs_rebalance(
        self,
        symbol: str,
        current_weight: float,
        target_signal: Signal,
    ) -> bool:
        """
        Return True if |current_weight − target| exceeds rebalance_threshold.

        Prevents churn from minor probability fluctuations.

        Parameters
        ----------
        symbol : str
        current_weight : float
            Current portfolio weight of `symbol` (0.0 – 1.0).
        target_signal : Signal
            Latest signal for `symbol`.

        Returns
        -------
        bool
        """
        drift = abs(current_weight - target_signal.position_size_pct)
        if drift > self.rebalance_threshold:
            logger.debug(
                "Rebalance needed for %s: current=%.3f  target=%.3f  drift=%.3f",
                symbol,
                current_weight,
                target_signal.position_size_pct,
                drift,
            )
            return True
        return False

    def get_strategy_for_regime(self, regime_id: int) -> Optional[BaseStrategy]:
        """Return the strategy assigned to `regime_id`, or None."""
        return self._regime_to_strategy.get(regime_id)

    def get_vol_rank(self, regime_id: int) -> Optional[float]:
        """Return the vol_rank (0.0–1.0) for `regime_id`, or None."""
        return self._vol_ranks.get(regime_id)

    # ── Internal ──────────────────────────────────────────────────────

    def _make_strategy(self, position: float) -> BaseStrategy:
        """Instantiate the correct strategy for a given vol_rank position."""
        kwargs: dict = dict(
            min_confidence=self.min_confidence,
            uncertainty_size_mult=self.uncertainty_size_mult,
            ema_window=self.ema_window,
            atr_window=self.atr_window,
        )
        if position <= _LOW_VOL_THRESHOLD:
            return LowVolBullStrategy(**kwargs)
        elif position >= _HIGH_VOL_THRESHOLD:
            return HighVolDefensiveStrategy(**kwargs)
        else:
            return MidVolCautiousStrategy(**kwargs)

    def _get_strategy(
        self,
        regime_id: int,
        label: RegimeLabel = RegimeLabel.UNKNOWN,
    ) -> BaseStrategy:
        """
        Return the strategy for regime_id.

        Falls back to LABEL_TO_STRATEGY if regime_id is unknown (e.g. after
        HMM retraining before update_regime_infos() is called).
        """
        strategy = self._regime_to_strategy.get(regime_id)
        if strategy is not None:
            return strategy

        # Fallback: use label-based mapping
        strategy_cls = LABEL_TO_STRATEGY.get(label, HighVolDefensiveStrategy)
        logger.warning(
            "No strategy for regime_id=%d ('%s') — falling back to %s",
            regime_id,
            label.value,
            strategy_cls.__name__,
        )
        return strategy_cls(
            min_confidence=self.min_confidence,
            uncertainty_size_mult=self.uncertainty_size_mult,
            ema_window=self.ema_window,
            atr_window=self.atr_window,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy RegimeStrategy (kept for backward compatibility with signal_generator
# and existing tests that import AllocationDecision + RegimeStrategy)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeStrategy:
    """
    Legacy allocation interface used by SignalGenerator and existing tests.

    Maps a RegimeResult (with a volatility Regime enum) to an AllocationDecision.
    New code should prefer StrategyOrchestrator + Signal.

    Parameters
    ----------
    low_vol_allocation : float
    mid_vol_allocation_trend : float
    mid_vol_allocation_no_trend : float
    high_vol_allocation : float
    low_vol_leverage : float
    mid_vol_leverage_trend : float
    rebalance_threshold : float
    uncertainty_size_mult : float
    min_confidence : float
    """

    def __init__(
        self,
        low_vol_allocation: float = 0.95,
        mid_vol_allocation_trend: float = 0.95,
        mid_vol_allocation_no_trend: float = 0.60,
        high_vol_allocation: float = 0.60,
        low_vol_leverage: float = 1.25,
        mid_vol_leverage_trend: float = 1.6,
        rebalance_threshold: float = 0.10,
        uncertainty_size_mult: float = 0.50,
        min_confidence: float = 0.55,
    ) -> None:
        self.low_vol_allocation = low_vol_allocation
        self.mid_vol_allocation_trend = mid_vol_allocation_trend
        self.mid_vol_allocation_no_trend = mid_vol_allocation_no_trend
        self.high_vol_allocation = high_vol_allocation
        self.low_vol_leverage = low_vol_leverage
        self.mid_vol_leverage_trend = mid_vol_leverage_trend
        self.rebalance_threshold = rebalance_threshold
        self.uncertainty_size_mult = uncertainty_size_mult
        self.min_confidence = min_confidence

    def get_allocation(
        self,
        regime_result: RegimeResult,
        trend_signal: Optional[bool] = None,
    ) -> AllocationDecision:
        """
        Compute target allocation and leverage for one bar.

        Parameters
        ----------
        regime_result : RegimeResult
        trend_signal : bool | None
            True = uptrend confirmed, False/None = no trend (mid-vol only).

        Returns
        -------
        AllocationDecision
        """
        alloc, leverage = self._base_allocation(regime_result.regime, trend_signal)
        alloc_scaled, was_scaled = self._apply_confidence_scaling(
            alloc, regime_result.confidence
        )
        if was_scaled:
            leverage = 1.0  # No leverage when uncertain

        notes = self._build_notes(regime_result.regime, trend_signal, was_scaled)
        return AllocationDecision(
            regime=regime_result.regime,
            allocation_fraction=round(alloc_scaled, 4),
            leverage=round(leverage, 4),
            confidence_scaled=was_scaled,
            notes=notes,
        )

    def needs_rebalance(
        self,
        current_weight: float,
        target_weight: float,
    ) -> bool:
        """Return True if drift exceeds rebalance_threshold."""
        return abs(current_weight - target_weight) > self.rebalance_threshold

    # ── Internal ──────────────────────────────────────────────────────

    def _base_allocation(
        self,
        regime: Regime,
        trend_signal: Optional[bool],
    ) -> tuple[float, float]:
        """Return (allocation_fraction, leverage) before confidence scaling."""
        if regime == Regime.LOW_VOL:
            return self.low_vol_allocation, self.low_vol_leverage

        if regime == Regime.HIGH_VOL:
            return self.high_vol_allocation, 1.0

        if regime == Regime.MID_VOL:
            if trend_signal:
                return self.mid_vol_allocation_trend, self.mid_vol_leverage_trend
            return self.mid_vol_allocation_no_trend, 1.0

        # UNKNOWN — defensive default
        return self.high_vol_allocation * self.uncertainty_size_mult, 1.0

    def _apply_confidence_scaling(
        self,
        allocation: float,
        confidence: float,
    ) -> tuple[float, bool]:
        """Scale down if confidence < min_confidence."""
        if confidence < self.min_confidence:
            return round(allocation * self.uncertainty_size_mult, 4), True
        return allocation, False

    def _build_notes(
        self,
        regime: Regime,
        trend_signal: Optional[bool],
        was_scaled: bool,
    ) -> str:
        parts = [f"regime={regime.value}"]
        if regime == Regime.MID_VOL:
            parts.append(f"trend={'up' if trend_signal else 'down/unknown'}")
        if was_scaled:
            parts.append("confidence_scaled=True [size halved]")
        return "  ".join(parts)
