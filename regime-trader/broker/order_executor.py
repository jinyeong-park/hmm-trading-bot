"""
order_executor.py — Order placement, modification, and cancellation.

Responsibilities:
  - Translate TradingSignal objects into Alpaca market/limit orders.
  - Support bracket orders (entry + stop-loss + take-profit).
  - Track order lifecycle: submitted → partial fill → filled / cancelled.
  - Cancel stale open orders that exceed a time-in-force threshold.
  - Enforce order deduplication (no double-entry for the same signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from broker.alpaca_client import AlpacaClient
from core.signal_generator import TradingSignal


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Internal representation of a submitted order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    broker_response: dict[str, Any] = field(default_factory=dict)


class OrderExecutor:
    """
    Places and manages orders through the Alpaca brokerage.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client.
    use_limit_orders : bool
        If True, submit limit orders at the current ask/bid instead of market.
    limit_slippage_pct : float
        Maximum allowed slippage for limit order pricing (fraction of price).
    stale_order_minutes : int
        Cancel unfilled orders older than this threshold in minutes.
    """

    def __init__(
        self,
        client: AlpacaClient,
        use_limit_orders: bool = False,
        limit_slippage_pct: float = 0.001,
        stale_order_minutes: int = 30,
    ) -> None:
        self.client = client
        self.use_limit_orders = use_limit_orders
        self.limit_slippage_pct = limit_slippage_pct
        self.stale_order_minutes = stale_order_minutes

        self._open_orders: dict[str, Order] = {}   # order_id → Order
        self._pending_symbols: set[str] = set()    # symbols with open orders

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_signal(self, signal: TradingSignal, equity: float) -> Optional[Order]:
        """
        Convert a TradingSignal into a submitted order.

        Parameters
        ----------
        signal : TradingSignal
        equity : float
            Current portfolio equity (used to compute share qty from weight).

        Returns
        -------
        Order | None
            Submitted order, or None if signal was HOLD or execution was skipped.
        """
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by ID.

        Returns
        -------
        bool
            True if cancellation was accepted by the broker.
        """
        raise NotImplementedError

    def cancel_all(self) -> list[str]:
        """
        Cancel all open orders tracked by this executor.

        Returns
        -------
        list[str]
            Order IDs that were successfully cancelled.
        """
        raise NotImplementedError

    def cancel_stale_orders(self) -> list[str]:
        """
        Cancel any open orders that have exceeded stale_order_minutes.

        Returns
        -------
        list[str]
            Order IDs that were cancelled.
        """
        raise NotImplementedError

    def sync_order_statuses(self) -> None:
        """
        Query Alpaca for the latest status of all tracked open orders and update
        internal state. Should be called at the start of each trading loop iteration.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_order_params(
        self, signal: TradingSignal, qty: float
    ) -> dict[str, Any]:
        """
        Construct Alpaca API order parameters dict from a signal.

        Parameters
        ----------
        signal : TradingSignal
        qty : float
            Resolved share quantity.

        Returns
        -------
        dict[str, Any]
        """
        raise NotImplementedError

    def _compute_qty(self, weight: float, price: float, equity: float) -> float:
        """
        Convert target portfolio weight to share quantity.

        Parameters
        ----------
        weight : float
            Target allocation weight (0.0–1.0).
        price : float
            Current price per share.
        equity : float
            Total portfolio equity.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _is_duplicate(self, symbol: str) -> bool:
        """Return True if there is already an open or pending order for `symbol`."""
        raise NotImplementedError
