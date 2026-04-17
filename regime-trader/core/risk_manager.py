"""
risk_manager.py — Position sizing, leverage enforcement, and drawdown circuit breakers.

The risk manager operates INDEPENDENTLY of the HMM.  Even if the HMM fails
completely, circuit breakers catch drawdowns based on actual P&L.
Defense in depth.  The risk manager has ABSOLUTE VETO POWER over any signal.

Key design decisions:
  - CircuitBreaker tracks all trigger events for post-trade audit.
  - Peak-DD halt writes a `trading_halted.lock` file that requires MANUAL
    deletion to resume — this is intentional.
  - Duplicate-order suppression is keyed by (symbol, direction) with a 60s TTL.
  - Correlation check reduces/rejects based on 60-day rolling correlation
    against existing positions, passed in via PortfolioState.price_history.
  - validate_signal() is the PRIMARY entry point.  check_order() is kept for
    backward compat with the backtester.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Path written on peak-DD halt; must be deleted manually to resume.
_HALT_LOCK_FILE = Path("trading_halted.lock")

# Minimum notional value of a position.
_MIN_POSITION_USD = 100.0

# Duplicate-order TTL in seconds.
_DUPLICATE_TTL_SECONDS = 60

# Overnight gap-risk factor: assume price can gap 3× the stop distance.
_GAP_MULTIPLIER = 3.0

# Max portfolio loss assumed from a gap event.
_GAP_MAX_PORTFOLIO_LOSS = 0.02  # 2 %


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class RiskAction(str, Enum):
    """Possible responses from the risk manager."""

    ALLOW = "allow"
    REDUCE = "reduce"   # Scale position size down
    HALT = "halt"       # Block new entries
    REJECT = "reject"   # Single-trade limit exceeded


class CircuitBreakerType(str, Enum):
    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskCheckResult:
    """Outcome of a risk validation call (legacy / backtester interface)."""

    action: RiskAction
    size_multiplier: float = 1.0    # Applied to raw position size
    reason: str = ""


@dataclass
class DrawdownState:
    """Rolling drawdown tracking state."""

    peak_equity: float = 0.0
    daily_open_equity: float = 0.0
    weekly_open_equity: float = 0.0
    current_equity: float = 0.0
    daily_trades: int = 0
    last_reset_date: Optional[date] = None
    last_weekly_reset: Optional[date] = None
    halted_until: Optional[datetime] = None


@dataclass
class Position:
    """A single open position in the portfolio."""

    symbol: str
    shares: float
    avg_entry_price: float
    current_price: float
    stop_price: float
    sector: str = "UNKNOWN"

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_entry_price)


@dataclass
class PortfolioState:
    """Complete snapshot of the portfolio at a given moment."""

    equity: float                         # Total portfolio value (cash + positions)
    cash: float                           # Free cash
    buying_power: float                   # Margin-adjusted buying power
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0                # P&L from start of current trading day
    weekly_pnl: float = 0.0               # P&L from start of current trading week
    peak_equity: float = 0.0              # Highest equity ever reached
    current_drawdown: float = 0.0         # Peak-to-trough DD as positive fraction
    circuit_breaker_status: dict[str, bool] = field(default_factory=dict)
    flicker_rate: float = 0.0             # Regime transitions per flicker_window bars
    # Keyed by symbol → pd.Series of closes (for correlation checks)
    price_history: dict[str, pd.Series] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class RiskModification:
    """A single modification applied to a signal by the risk manager."""

    field: str          # e.g. "position_size_pct", "leverage"
    original: object
    modified: object
    reason: str


@dataclass
class RiskDecision:
    """Full outcome of validate_signal()."""

    approved: bool
    modified_signal: object             # The (possibly modified) Signal, or None if rejected
    rejection_reason: str = ""
    modifications: list[RiskModification] = field(default_factory=list)
    size_multiplier: float = 1.0        # Net size scaling applied
    warnings: list[str] = field(default_factory=list)


@dataclass
class CircuitBreakerEvent:
    """Logged entry for every circuit-breaker trigger."""

    timestamp: datetime
    breaker_type: CircuitBreakerType
    actual_dd: float                    # Drawdown that triggered the breaker
    equity: float
    positions_closed: list[str]         # Symbols force-closed (empty if reduce-only)
    hmm_regime: str                     # Regime label at time of trigger
    hmm_was_wrong: Optional[bool]       # Set during post-trade analysis


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Tracks daily / weekly / peak drawdown and emits REDUCE or HALT actions.

    Parameters
    ----------
    daily_dd_reduce, daily_dd_halt : float
        Thresholds as positive fractions (e.g. 0.02 = 2 %).
    weekly_dd_reduce, weekly_dd_halt : float
    max_dd_from_peak : float
    halt_lock_path : Path
        Where to write the manual-resume lock file on peak halt.
    """

    def __init__(
        self,
        daily_dd_reduce: float = 0.02,
        daily_dd_halt: float = 0.03,
        weekly_dd_reduce: float = 0.05,
        weekly_dd_halt: float = 0.07,
        max_dd_from_peak: float = 0.10,
        halt_lock_path: Path = _HALT_LOCK_FILE,
    ) -> None:
        self.daily_dd_reduce = daily_dd_reduce
        self.daily_dd_halt = daily_dd_halt
        self.weekly_dd_reduce = weekly_dd_reduce
        self.weekly_dd_halt = weekly_dd_halt
        self.max_dd_from_peak = max_dd_from_peak
        self.halt_lock_path = halt_lock_path

        self._peak_equity: float = 0.0
        self._daily_open: float = 0.0
        self._weekly_open: float = 0.0
        self._current_equity: float = 0.0
        self._daily_halted: bool = False
        self._weekly_halted: bool = False
        self._peak_halted: bool = False
        self._history: list[CircuitBreakerEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, equity: float, timestamp: datetime) -> None:
        """Record new equity; used only for internal state — does not check breakers."""
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        if self._daily_open == 0.0:
            self._daily_open = equity
        if self._weekly_open == 0.0:
            self._weekly_open = equity
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def reset_daily(self, today: date) -> None:
        """Call at the start of each trading day."""
        self._daily_open = self._current_equity
        self._daily_halted = False
        logger.debug("CircuitBreaker: daily reset on %s  open_equity=%.2f", today, self._daily_open)

    def reset_weekly(self, week_start: date) -> None:
        """Call at the start of each trading week."""
        self._weekly_open = self._current_equity
        self._weekly_halted = False
        logger.debug("CircuitBreaker: weekly reset on %s  open_equity=%.2f", week_start, self._weekly_open)

    def check(
        self,
        portfolio_state: Optional[PortfolioState] = None,
        hmm_regime: str = "UNKNOWN",
    ) -> RiskCheckResult:
        """
        Evaluate all drawdown limits and return the most restrictive action.

        Precedence: peak_halt > daily_halt > weekly_halt > reduce > allow.
        Logs every new trigger.
        """
        # 1. Check lock file (manual halt in place)
        if self.halt_lock_path.exists():
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"trading_halted.lock exists — delete {self.halt_lock_path} to resume",
            )

        # 2. Peak drawdown halt
        peak_dd = self._dd_from_peak()
        if peak_dd >= self.max_dd_from_peak:
            if not self._peak_halted:
                self._peak_halted = True
                self._log_trigger(
                    CircuitBreakerType.PEAK_HALT,
                    peak_dd, hmm_regime, portfolio_state,
                )
                self._write_halt_lock(peak_dd)
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Peak drawdown {peak_dd:.1%} ≥ {self.max_dd_from_peak:.1%} — HALTED",
            )

        daily_dd = self._dd_daily()
        weekly_dd = self._dd_weekly()

        # 3. Daily halt
        if daily_dd >= self.daily_dd_halt or self._daily_halted:
            if not self._daily_halted:
                self._daily_halted = True
                self._log_trigger(
                    CircuitBreakerType.DAILY_HALT,
                    daily_dd, hmm_regime, portfolio_state,
                )
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Daily drawdown {daily_dd:.1%} ≥ {self.daily_dd_halt:.1%} — halted for day",
            )

        # 4. Weekly halt
        if weekly_dd >= self.weekly_dd_halt or self._weekly_halted:
            if not self._weekly_halted:
                self._weekly_halted = True
                self._log_trigger(
                    CircuitBreakerType.WEEKLY_HALT,
                    weekly_dd, hmm_regime, portfolio_state,
                )
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Weekly drawdown {weekly_dd:.1%} ≥ {self.weekly_dd_halt:.1%} — halted for week",
            )

        # 5. Reduce (most restrictive reduce wins)
        if daily_dd >= self.daily_dd_reduce:
            self._log_trigger(
                CircuitBreakerType.DAILY_REDUCE,
                daily_dd, hmm_regime, portfolio_state, first_only=True,
            )
            return RiskCheckResult(
                action=RiskAction.REDUCE,
                size_multiplier=0.50,
                reason=f"Daily drawdown {daily_dd:.1%} ≥ {self.daily_dd_reduce:.1%} — sizes halved",
            )

        if weekly_dd >= self.weekly_dd_reduce:
            self._log_trigger(
                CircuitBreakerType.WEEKLY_REDUCE,
                weekly_dd, hmm_regime, portfolio_state, first_only=True,
            )
            return RiskCheckResult(
                action=RiskAction.REDUCE,
                size_multiplier=0.50,
                reason=f"Weekly drawdown {weekly_dd:.1%} ≥ {self.weekly_dd_reduce:.1%} — sizes halved",
            )

        return RiskCheckResult(action=RiskAction.ALLOW, size_multiplier=1.0)

    def get_history(self) -> list[CircuitBreakerEvent]:
        return list(self._history)

    def any_active(self) -> bool:
        """True if any halt or reduce breaker is currently active."""
        if self.halt_lock_path.exists():
            return True
        return self._peak_halted or self._daily_halted or self._weekly_halted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dd_from_peak(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity)

    def _dd_daily(self) -> float:
        if self._daily_open <= 0:
            return 0.0
        return max(0.0, (self._daily_open - self._current_equity) / self._daily_open)

    def _dd_weekly(self) -> float:
        if self._weekly_open <= 0:
            return 0.0
        return max(0.0, (self._weekly_open - self._current_equity) / self._weekly_open)

    def _write_halt_lock(self, dd: float) -> None:
        try:
            self.halt_lock_path.write_text(
                f"Trading halted: peak drawdown {dd:.2%}\n"
                "Delete this file to resume trading.\n"
            )
            logger.critical(
                "PEAK DRAWDOWN HALT: %.1f%% — wrote %s",
                dd * 100, self.halt_lock_path,
            )
        except OSError as exc:
            logger.error("Could not write halt lock file: %s", exc)

    def _log_trigger(
        self,
        breaker_type: CircuitBreakerType,
        actual_dd: float,
        hmm_regime: str,
        portfolio_state: Optional[PortfolioState],
        first_only: bool = False,
    ) -> None:
        # For reduce breakers, avoid flooding the log; log once per session
        if first_only:
            already = any(
                e.breaker_type == breaker_type for e in self._history
            )
            if already:
                return

        positions_closed: list[str] = []
        equity = self._current_equity
        if portfolio_state is not None:
            equity = portfolio_state.equity
            if breaker_type in (CircuitBreakerType.DAILY_HALT, CircuitBreakerType.WEEKLY_HALT, CircuitBreakerType.PEAK_HALT):
                positions_closed = list(portfolio_state.positions.keys())

        event = CircuitBreakerEvent(
            timestamp=datetime.now(tz=timezone.utc),
            breaker_type=breaker_type,
            actual_dd=actual_dd,
            equity=equity,
            positions_closed=positions_closed,
            hmm_regime=hmm_regime,
            hmm_was_wrong=None,
        )
        self._history.append(event)
        logger.warning(
            "CIRCUIT BREAKER [%s]: dd=%.2f%% equity=%.2f regime=%s positions=%s",
            breaker_type.value, actual_dd * 100, equity, hmm_regime, positions_closed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main RiskManager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Enforces all risk constraints before orders are submitted.

    Parameters
    ----------
    max_risk_per_trade : float
        Max equity fraction risked on a single position (1%).
    max_exposure : float
        Max gross exposure as a fraction of equity (80%).
    max_leverage : float
        Hard leverage ceiling (1.25×).
    max_single_position : float
        Max weight of any single asset (15%).
    max_correlated_exposure : float
        Max weight in one sector (30%).
    max_concurrent : int
        Max simultaneous open positions (5).
    max_daily_trades : int
        Hard cap on orders per calendar day (20).
    daily_dd_reduce, daily_dd_halt : float
        Daily drawdown thresholds.
    weekly_dd_reduce, weekly_dd_halt : float
        Weekly drawdown thresholds.
    max_dd_from_peak : float
        Rolling peak-to-trough halt threshold (10%).
    correlation_reduce_threshold : float
        Correlation above which new position size is halved (0.70).
    correlation_reject_threshold : float
        Correlation above which the trade is rejected (0.85).
    halt_lock_path : Path
        Where to write the peak-DD halt lock file.
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.01,
        max_exposure: float = 0.80,
        max_leverage: float = 1.25,
        max_single_position: float = 0.15,
        max_correlated_exposure: float = 0.30,
        max_concurrent: int = 5,
        max_daily_trades: int = 20,
        daily_dd_reduce: float = 0.02,
        daily_dd_halt: float = 0.03,
        weekly_dd_reduce: float = 0.05,
        weekly_dd_halt: float = 0.07,
        max_dd_from_peak: float = 0.10,
        correlation_reduce_threshold: float = 0.70,
        correlation_reject_threshold: float = 0.85,
        halt_lock_path: Path = _HALT_LOCK_FILE,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_exposure = max_exposure
        self.max_leverage = max_leverage
        self.max_single_position = max_single_position
        self.max_correlated_exposure = max_correlated_exposure
        self.max_concurrent = max_concurrent
        self.max_daily_trades = max_daily_trades
        self.daily_dd_reduce = daily_dd_reduce
        self.daily_dd_halt = daily_dd_halt
        self.weekly_dd_reduce = weekly_dd_reduce
        self.weekly_dd_halt = weekly_dd_halt
        self.max_dd_from_peak = max_dd_from_peak
        self.correlation_reduce_threshold = correlation_reduce_threshold
        self.correlation_reject_threshold = correlation_reject_threshold

        self._state: DrawdownState = DrawdownState()
        self.circuit_breaker = CircuitBreaker(
            daily_dd_reduce=daily_dd_reduce,
            daily_dd_halt=daily_dd_halt,
            weekly_dd_reduce=weekly_dd_reduce,
            weekly_dd_halt=weekly_dd_halt,
            max_dd_from_peak=max_dd_from_peak,
            halt_lock_path=halt_lock_path,
        )

        # (symbol, direction) → last order timestamp for duplicate suppression
        self._recent_orders: dict[tuple[str, str], datetime] = {}

    # ------------------------------------------------------------------
    # Primary entry point (live trading)
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal,                         # core.regime_strategies.Signal
        portfolio_state: PortfolioState,
        is_overnight: bool = False,
        bid_ask_spread_pct: Optional[float] = None,
    ) -> RiskDecision:
        """
        Full signal validation with ABSOLUTE VETO POWER.

        Checks (in order — first rejection wins):
          1. Lock file / halt
          2. Stop loss present
          3. Circuit breakers (DD-based)
          4. Duplicate order suppression
          5. Max daily trades
          6. Leverage enforcement
          7. Position size computation + gap risk
          8. Portfolio exposure limits
          9. Max concurrent positions
         10. Bid-ask spread
         11. Buying power
         12. Correlation against existing positions

        Returns
        -------
        RiskDecision
            approved=False with rejection_reason, or approved=True with
            possibly-modified signal.
        """
        import copy
        modified = copy.copy(signal)
        mods: list[RiskModification] = []
        warnings: list[str] = []
        size_mult = 1.0

        equity = portfolio_state.equity
        regime_name = getattr(signal, "regime_name", "UNKNOWN") or "UNKNOWN"

        # ── 1. Halt lock / circuit breaker ───────────────────────────
        cb_result = self.circuit_breaker.check(portfolio_state, hmm_regime=regime_name)
        if cb_result.action == RiskAction.HALT:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=cb_result.reason,
            )
        if cb_result.action == RiskAction.REDUCE:
            size_mult *= cb_result.size_multiplier
            mods.append(RiskModification(
                field="position_size_pct",
                original=signal.position_size_pct,
                modified=signal.position_size_pct * size_mult,
                reason=cb_result.reason,
            ))
            warnings.append(cb_result.reason)

        # ── 2. Stop loss required ────────────────────────────────────
        stop = getattr(signal, "stop_loss", None)
        entry = getattr(signal, "entry_price", None)
        if stop is None or entry is None or stop <= 0 or entry <= 0:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason="Order rejected: stop_loss or entry_price missing/invalid",
            )
        if abs(entry - stop) < 1e-8:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason="Order rejected: stop_loss == entry_price (zero risk distance)",
            )

        # ── 3. Duplicate suppression ──────────────────────────────────
        direction = str(getattr(signal, "direction", "LONG"))
        dup_key = (signal.symbol, direction)
        now = portfolio_state.timestamp or datetime.now(tz=timezone.utc)
        if dup_key in self._recent_orders:
            age = (now - self._recent_orders[dup_key]).total_seconds()
            if age < _DUPLICATE_TTL_SECONDS:
                return RiskDecision(
                    approved=False,
                    modified_signal=None,
                    rejection_reason=(
                        f"Duplicate order suppressed: {signal.symbol} {direction} "
                        f"sent {age:.0f}s ago (TTL={_DUPLICATE_TTL_SECONDS}s)"
                    ),
                )

        # ── 4. Max daily trades ───────────────────────────────────────
        if self._state.daily_trades >= self.max_daily_trades:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=(
                    f"Daily trade limit reached: {self._state.daily_trades}/{self.max_daily_trades}"
                ),
            )

        # ── 5. Leverage enforcement ───────────────────────────────────
        requested_lev = getattr(signal, "leverage", 1.0)
        n_open = len(portfolio_state.positions)
        regime_uncertain = getattr(signal, "regime_name", "") in ("UNKNOWN", "")
        cb_active = self.circuit_breaker.any_active()
        high_flicker = portfolio_state.flicker_rate > 0.20  # >20% transitions
        force_1x = (
            regime_uncertain
            or cb_active
            or n_open >= 3
            or high_flicker
        )
        max_allowed_leverage = (
            self.max_leverage if not force_1x else 1.0
        )
        if requested_lev > max_allowed_leverage:
            mods.append(RiskModification(
                field="leverage",
                original=requested_lev,
                modified=max_allowed_leverage,
                reason=(
                    f"Leverage forced to {max_allowed_leverage}× "
                    f"(uncertain={regime_uncertain}, cb_active={cb_active}, "
                    f"n_open={n_open}, high_flicker={high_flicker})"
                ),
            ))
            modified.leverage = max_allowed_leverage
            warnings.append(f"Leverage reduced from {requested_lev}× to {max_allowed_leverage}×")

        # ── 6. Position size computation ──────────────────────────────
        stop_dist = abs(entry - stop)
        raw_shares = (equity * self.max_risk_per_trade) / stop_dist

        # Cap by allocation fraction and by single-position limit
        alloc_frac = getattr(signal, "position_size_pct", self.max_single_position)
        max_notional_alloc = equity * alloc_frac
        max_notional_pos = equity * self.max_single_position
        max_shares_alloc = max_notional_alloc / entry
        max_shares_pos = max_notional_pos / entry

        capped_shares = min(raw_shares, max_shares_alloc, max_shares_pos)

        # Gap risk for overnight positions
        if is_overnight:
            gap_dist = _GAP_MULTIPLIER * stop_dist
            max_shares_gap = (equity * _GAP_MAX_PORTFOLIO_LOSS) / gap_dist
            if max_shares_gap < capped_shares:
                mods.append(RiskModification(
                    field="shares",
                    original=round(capped_shares, 4),
                    modified=round(max_shares_gap, 4),
                    reason=(
                        f"Overnight gap-risk cap: 3× stop_dist={stop_dist:.4f} "
                        f"→ max loss {_GAP_MAX_PORTFOLIO_LOSS:.0%} of portfolio"
                    ),
                ))
                capped_shares = max_shares_gap

        # Apply circuit-breaker size multiplier
        capped_shares *= size_mult

        # Minimum position check
        if capped_shares * entry < _MIN_POSITION_USD:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=(
                    f"Position too small: {capped_shares:.4f} × {entry:.2f} "
                    f"= ${capped_shares * entry:.2f} < ${_MIN_POSITION_USD}"
                ),
            )

        # Sync modified signal
        new_size_pct = (capped_shares * entry) / equity
        if abs(new_size_pct - alloc_frac) > 1e-6:
            mods.append(RiskModification(
                field="position_size_pct",
                original=round(alloc_frac, 6),
                modified=round(new_size_pct, 6),
                reason="Adjusted by risk sizing rules",
            ))
            modified.position_size_pct = new_size_pct

        # ── 7. Portfolio exposure limits ──────────────────────────────
        current_notional = sum(p.notional for p in portfolio_state.positions.values())
        new_notional = capped_shares * entry
        total_notional = current_notional + new_notional

        if total_notional > equity * self.max_exposure:
            over_by = total_notional - equity * self.max_exposure
            reduced_notional = new_notional - over_by
            if reduced_notional < _MIN_POSITION_USD:
                return RiskDecision(
                    approved=False,
                    modified_signal=None,
                    rejection_reason=(
                        f"Max exposure {self.max_exposure:.0%} would be breached: "
                        f"current={current_notional:.0f} + new={new_notional:.0f} "
                        f"> {equity * self.max_exposure:.0f}"
                    ),
                )
            reduced_shares = reduced_notional / entry
            reduced_size_pct = reduced_notional / equity
            mods.append(RiskModification(
                field="position_size_pct",
                original=round(new_size_pct, 6),
                modified=round(reduced_size_pct, 6),
                reason=f"Reduced to stay within max_exposure={self.max_exposure:.0%}",
            ))
            modified.position_size_pct = reduced_size_pct
            capped_shares = reduced_shares
            warnings.append(f"Position reduced to respect max_exposure={self.max_exposure:.0%}")

        # ── 8. Max concurrent positions ───────────────────────────────
        if signal.symbol not in portfolio_state.positions:
            if n_open >= self.max_concurrent:
                return RiskDecision(
                    approved=False,
                    modified_signal=None,
                    rejection_reason=(
                        f"Max concurrent positions reached: "
                        f"{n_open}/{self.max_concurrent}"
                    ),
                )

        # ── 9. Bid-ask spread check ───────────────────────────────────
        if bid_ask_spread_pct is not None and bid_ask_spread_pct > 0.005:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=(
                    f"Bid-ask spread {bid_ask_spread_pct:.3%} > 0.5% for {signal.symbol}"
                ),
            )

        # ── 10. Buying power check ────────────────────────────────────
        required_bp = capped_shares * entry
        if required_bp > portfolio_state.buying_power:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=(
                    f"Insufficient buying power: need ${required_bp:.2f}, "
                    f"have ${portfolio_state.buying_power:.2f}"
                ),
            )

        # ── 11. Correlation check ─────────────────────────────────────
        corr_result = self._check_correlation(
            signal.symbol,
            portfolio_state.price_history,
            portfolio_state.positions,
        )
        if corr_result == "reject":
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=(
                    f"Correlation with existing positions > "
                    f"{self.correlation_reject_threshold:.0%} — trade rejected"
                ),
            )
        if corr_result == "reduce":
            old_pct = modified.position_size_pct
            modified.position_size_pct = old_pct * 0.50
            mods.append(RiskModification(
                field="position_size_pct",
                original=round(old_pct, 6),
                modified=round(modified.position_size_pct, 6),
                reason=(
                    f"High correlation with existing positions "
                    f"(>{self.correlation_reduce_threshold:.0%}) — size halved"
                ),
            ))
            warnings.append("Size halved due to high portfolio correlation")

        # ── Approved ──────────────────────────────────────────────────
        self._recent_orders[dup_key] = now

        if mods:
            logger.info(
                "Signal approved with %d modification(s) for %s", len(mods), signal.symbol
            )
        else:
            logger.debug("Signal approved: %s %s", signal.symbol, direction)

        return RiskDecision(
            approved=True,
            modified_signal=modified,
            modifications=mods,
            size_multiplier=size_mult,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Backward-compatible interface (used by backtester)
    # ------------------------------------------------------------------

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """
        Update current equity and advance daily/weekly windows if needed.

        Parameters
        ----------
        equity : float
        timestamp : datetime
        """
        s = self._state
        today = timestamp.date()

        # Initialise on first call
        if s.peak_equity == 0.0:
            s.peak_equity = equity
            s.daily_open_equity = equity
            s.weekly_open_equity = equity
            s.current_equity = equity
            s.last_reset_date = today
            s.last_weekly_reset = today
            self.circuit_breaker.update(equity, timestamp)
            return

        # Daily reset
        if s.last_reset_date != today:
            self._reset_daily_counters(today)

        # Weekly reset (Monday = weekday 0)
        if today.weekday() == 0 and s.last_weekly_reset != today:
            self._reset_weekly_counters(today)

        s.current_equity = equity
        s.peak_equity = max(s.peak_equity, equity)

        # Mirror state into the CircuitBreaker
        self.circuit_breaker._current_equity = equity
        self.circuit_breaker._peak_equity = s.peak_equity
        self.circuit_breaker._daily_open = s.daily_open_equity
        self.circuit_breaker._weekly_open = s.weekly_open_equity

        logger.debug(
            "update_equity: equity=%.2f  daily_dd=%.2f%%  weekly_dd=%.2f%%  peak_dd=%.2f%%",
            equity,
            self._current_drawdown_daily() * 100,
            self._current_drawdown_weekly() * 100,
            self._current_drawdown_from_peak() * 100,
        )

    def check_order(
        self,
        symbol: str,
        proposed_notional: float,
        current_positions: dict[str, float],
        equity: float,
    ) -> RiskCheckResult:
        """
        Validate a proposed order against all active risk limits.

        Parameters
        ----------
        symbol : str
        proposed_notional : float
            Dollar notional of the proposed order.
        current_positions : dict[str, float]
            symbol → current notional.
        equity : float

        Returns
        -------
        RiskCheckResult
        """
        # Circuit breaker (no portfolio_state here — use internal state)
        cb = self.circuit_breaker.check()
        if cb.action in (RiskAction.HALT, RiskAction.REJECT):
            return cb

        # Max daily trades
        if self._state.daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                action=RiskAction.REJECT,
                size_multiplier=0.0,
                reason=f"Daily trade limit: {self._state.daily_trades}/{self.max_daily_trades}",
            )

        # Max single position
        if proposed_notional > equity * self.max_single_position:
            max_allowed = equity * self.max_single_position
            return RiskCheckResult(
                action=RiskAction.REJECT,
                size_multiplier=0.0,
                reason=(
                    f"Single position too large: ${proposed_notional:.0f} "
                    f"> max ${max_allowed:.0f} ({self.max_single_position:.0%} of equity)"
                ),
            )

        # Max exposure
        current_notional = sum(current_positions.values())
        if current_notional + proposed_notional > equity * self.max_exposure:
            return RiskCheckResult(
                action=RiskAction.REJECT,
                size_multiplier=0.0,
                reason=(
                    f"Max exposure breach: current={current_notional:.0f} + "
                    f"new={proposed_notional:.0f} > {equity * self.max_exposure:.0f}"
                ),
            )

        # Max concurrent
        if symbol not in current_positions and len(current_positions) >= self.max_concurrent:
            return RiskCheckResult(
                action=RiskAction.REJECT,
                size_multiplier=0.0,
                reason=f"Max concurrent positions: {len(current_positions)}/{self.max_concurrent}",
            )

        # Apply circuit-breaker size multiplier (REDUCE case passthrough)
        return RiskCheckResult(
            action=cb.action,
            size_multiplier=cb.size_multiplier,
            reason=cb.reason,
        )

    def compute_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        allocation_fraction: float,
        is_overnight: bool = False,
    ) -> float:
        """
        Calculate share quantity using risk-per-trade and ATR-based stop.

        Formula:
            shares = (equity × max_risk_per_trade) / |entry - stop|

        Capped by:
            1. allocation_fraction × equity / entry_price
            2. max_single_position × equity / entry_price
            3. Overnight gap-risk cap (if is_overnight=True)

        Parameters
        ----------
        equity : float
        entry_price : float
        stop_price : float
        allocation_fraction : float
        is_overnight : bool

        Returns
        -------
        float
            Number of shares (caller rounds as needed).

        Raises
        ------
        ValueError
            If entry_price == stop_price (undefined risk distance).
        """
        stop_dist = abs(entry_price - stop_price)
        if stop_dist < 1e-8:
            raise ValueError(
                f"entry_price ({entry_price}) == stop_price ({stop_price}): "
                "cannot compute position size with zero risk distance"
            )
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        # Risk-based size
        shares = (equity * self.max_risk_per_trade) / stop_dist

        # Cap by allocation and by single-position limit
        max_shares_alloc = (equity * allocation_fraction) / entry_price
        max_shares_pos = (equity * self.max_single_position) / entry_price
        shares = min(shares, max_shares_alloc, max_shares_pos)

        # Overnight gap-risk cap
        if is_overnight:
            gap_dist = _GAP_MULTIPLIER * stop_dist
            max_shares_gap = (equity * _GAP_MAX_PORTFOLIO_LOSS) / gap_dist
            shares = min(shares, max_shares_gap)

        return shares

    def register_trade(self) -> None:
        """Increment the intraday trade counter after a successful order fill."""
        self._state.daily_trades += 1
        logger.debug("Daily trade count: %d/%d", self._state.daily_trades, self.max_daily_trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_drawdown_daily(self) -> float:
        s = self._state
        if s.daily_open_equity <= 0:
            return 0.0
        return max(0.0, (s.daily_open_equity - s.current_equity) / s.daily_open_equity)

    def _current_drawdown_weekly(self) -> float:
        s = self._state
        if s.weekly_open_equity <= 0:
            return 0.0
        return max(0.0, (s.weekly_open_equity - s.current_equity) / s.weekly_open_equity)

    def _current_drawdown_from_peak(self) -> float:
        s = self._state
        if s.peak_equity <= 0:
            return 0.0
        return max(0.0, (s.peak_equity - s.current_equity) / s.peak_equity)

    def _drawdown_risk_action(self) -> RiskCheckResult:
        """Evaluate all drawdown limits; return most restrictive."""
        peak_dd = self._current_drawdown_from_peak()
        if peak_dd >= self.max_dd_from_peak:
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Peak drawdown {peak_dd:.1%} ≥ {self.max_dd_from_peak:.1%}",
            )
        daily_dd = self._current_drawdown_daily()
        if daily_dd >= self.daily_dd_halt:
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Daily drawdown {daily_dd:.1%} ≥ {self.daily_dd_halt:.1%}",
            )
        weekly_dd = self._current_drawdown_weekly()
        if weekly_dd >= self.weekly_dd_halt:
            return RiskCheckResult(
                action=RiskAction.HALT,
                size_multiplier=0.0,
                reason=f"Weekly drawdown {weekly_dd:.1%} ≥ {self.weekly_dd_halt:.1%}",
            )
        if daily_dd >= self.daily_dd_reduce:
            return RiskCheckResult(
                action=RiskAction.REDUCE,
                size_multiplier=0.50,
                reason=f"Daily drawdown {daily_dd:.1%} ≥ {self.daily_dd_reduce:.1%} — reduce 50%",
            )
        if weekly_dd >= self.weekly_dd_reduce:
            return RiskCheckResult(
                action=RiskAction.REDUCE,
                size_multiplier=0.50,
                reason=f"Weekly drawdown {weekly_dd:.1%} ≥ {self.weekly_dd_reduce:.1%} — reduce 50%",
            )
        return RiskCheckResult(action=RiskAction.ALLOW, size_multiplier=1.0)

    def _reset_daily_counters(self, today: date) -> None:
        s = self._state
        s.daily_open_equity = s.current_equity
        s.daily_trades = 0
        s.last_reset_date = today
        self.circuit_breaker.reset_daily(today)
        logger.debug("Daily counters reset: %s  open_equity=%.2f", today, s.daily_open_equity)

    def _reset_weekly_counters(self, today: date) -> None:
        s = self._state
        s.weekly_open_equity = s.current_equity
        s.last_weekly_reset = today
        self.circuit_breaker.reset_weekly(today)
        logger.debug("Weekly counters reset: %s  open_equity=%.2f", today, s.weekly_open_equity)

    def _check_correlation(
        self,
        symbol: str,
        price_history: dict[str, pd.Series],
        positions: dict[str, Position],
    ) -> str:
        """
        Return 'allow', 'reduce', or 'reject' based on 60-day rolling correlation
        of `symbol` against each existing position.
        """
        if symbol not in price_history or not positions:
            return "allow"

        target = price_history[symbol].pct_change().dropna()
        if len(target) < 60:
            return "allow"

        max_corr = 0.0
        for pos_sym, pos in positions.items():
            if pos_sym == symbol or pos_sym not in price_history:
                continue
            other = price_history[pos_sym].pct_change().dropna()
            common = target.index.intersection(other.index)
            if len(common) < 30:
                continue
            t = target.loc[common].iloc[-60:]
            o = other.loc[common].iloc[-60:]
            if t.std() == 0 or o.std() == 0:
                continue
            corr = float(t.corr(o))
            max_corr = max(max_corr, abs(corr))

        if max_corr >= self.correlation_reject_threshold:
            logger.warning(
                "Correlation reject: %s max_corr=%.3f ≥ %.2f",
                symbol, max_corr, self.correlation_reject_threshold,
            )
            return "reject"
        if max_corr >= self.correlation_reduce_threshold:
            logger.info(
                "Correlation reduce: %s max_corr=%.3f ≥ %.2f",
                symbol, max_corr, self.correlation_reduce_threshold,
            )
            return "reduce"
        return "allow"
