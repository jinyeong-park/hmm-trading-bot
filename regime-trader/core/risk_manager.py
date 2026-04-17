"""
risk_manager.py — Position sizing, leverage enforcement, and drawdown circuit breakers.

Responsibilities:
  - Compute per-trade position size from risk fraction and ATR-based stop distance.
  - Enforce max single-position weight, max concurrent positions, and max exposure.
  - Track intraday trade count and enforce max_daily_trades.
  - Monitor daily / weekly / peak-to-trough drawdown and emit halt/reduce signals.
  - Validate any proposed order against all active risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class RiskAction(str, Enum):
    """Possible responses from the risk manager."""

    ALLOW = "allow"
    REDUCE = "reduce"        # Scale position size down (drawdown reduce threshold hit)
    HALT = "halt"            # Block new entries (drawdown halt threshold hit)
    REJECT = "reject"        # Single-trade limit exceeded (max exposure, position cap)


@dataclass
class RiskCheckResult:
    """Outcome of a risk validation call."""

    action: RiskAction
    size_multiplier: float = 1.0    # Applied to raw position size (1.0 = unchanged)
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


class RiskManager:
    """
    Enforces all risk constraints before orders are submitted.

    Parameters
    ----------
    max_risk_per_trade : float
        Maximum equity fraction risked on a single position.
    max_exposure : float
        Maximum gross exposure as a fraction of equity.
    max_leverage : float
        Hard leverage ceiling.
    max_single_position : float
        Maximum weight of any single asset in the portfolio.
    max_concurrent : int
        Maximum number of simultaneous open positions.
    max_daily_trades : int
        Hard cap on orders placed per calendar day.
    daily_dd_reduce : float
        Daily drawdown threshold that triggers size reduction.
    daily_dd_halt : float
        Daily drawdown threshold that halts new entries.
    weekly_dd_reduce : float
        Weekly drawdown threshold that triggers size reduction.
    weekly_dd_halt : float
        Weekly drawdown threshold that halts entries for the week.
    max_dd_from_peak : float
        Rolling peak-to-trough drawdown limit that triggers full halt.
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.01,
        max_exposure: float = 0.80,
        max_leverage: float = 1.25,
        max_single_position: float = 0.15,
        max_concurrent: int = 5,
        max_daily_trades: int = 20,
        daily_dd_reduce: float = 0.02,
        daily_dd_halt: float = 0.03,
        weekly_dd_reduce: float = 0.05,
        weekly_dd_halt: float = 0.07,
        max_dd_from_peak: float = 0.10,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_exposure = max_exposure
        self.max_leverage = max_leverage
        self.max_single_position = max_single_position
        self.max_concurrent = max_concurrent
        self.max_daily_trades = max_daily_trades
        self.daily_dd_reduce = daily_dd_reduce
        self.daily_dd_halt = daily_dd_halt
        self.weekly_dd_reduce = weekly_dd_reduce
        self.weekly_dd_halt = weekly_dd_halt
        self.max_dd_from_peak = max_dd_from_peak

        self._state: DrawdownState = DrawdownState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """
        Update current equity and advance daily/weekly windows if needed.

        Parameters
        ----------
        equity : float
            Latest portfolio equity value in USD.
        timestamp : datetime
            Current bar timestamp (used for daily/weekly resets).
        """
        raise NotImplementedError

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
            Ticker being traded.
        proposed_notional : float
            Dollar notional value of the proposed order.
        current_positions : dict[str, float]
            Mapping of symbol → current notional value of open positions.
        equity : float
            Current portfolio equity.

        Returns
        -------
        RiskCheckResult
        """
        raise NotImplementedError

    def compute_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        allocation_fraction: float,
    ) -> float:
        """
        Calculate share quantity using the risk-per-trade and ATR-based stop.

        Parameters
        ----------
        equity : float
            Current portfolio equity.
        entry_price : float
            Proposed entry price per share.
        stop_price : float
            Stop-loss price per share.
        allocation_fraction : float
            Fraction of equity cleared for deployment by the strategy layer.

        Returns
        -------
        float
            Number of shares (may be fractional — caller rounds as needed).
        """
        raise NotImplementedError

    def register_trade(self) -> None:
        """Increment the intraday trade counter after a successful order fill."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_drawdown_daily(self) -> float:
        """Return daily drawdown as a positive fraction (0.0 = no drawdown)."""
        raise NotImplementedError

    def _current_drawdown_weekly(self) -> float:
        """Return weekly drawdown as a positive fraction."""
        raise NotImplementedError

    def _current_drawdown_from_peak(self) -> float:
        """Return peak-to-trough drawdown as a positive fraction."""
        raise NotImplementedError

    def _drawdown_risk_action(self) -> RiskCheckResult:
        """
        Evaluate all drawdown limits and return the most restrictive RiskCheckResult.

        Precedence: HALT > REDUCE > ALLOW
        """
        raise NotImplementedError

    def _reset_daily_counters(self, today: date) -> None:
        """Reset daily trade counter and record today's open equity."""
        raise NotImplementedError

    def _reset_weekly_counters(self, today: date) -> None:
        """Record this week's open equity for weekly drawdown calculation."""
        raise NotImplementedError
