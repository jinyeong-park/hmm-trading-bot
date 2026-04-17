"""
order_executor.py — Order placement, modification, and cancellation.

Responsibilities:
  - Translate Signal / TradingSignal objects into Alpaca limit/market orders.
  - Support OCO bracket orders (entry + stop-loss + take-profit).
  - Auto-cancel unfilled limit orders after 30 seconds; optionally retry at market.
  - Block duplicate orders (same symbol already has an open or pending order).
  - Enforce stop-tightening-only on modify_stop().
  - Emit a unique trade_id that links: signal → risk_decision → order → fill.

All alpaca-py imports are deferred so this module loads cleanly in backtest mode.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# Cancel unfilled limit orders after this many seconds
_LIMIT_CANCEL_SECONDS = 30

# Limit price is current price +/- this fraction for BUY / SELL
_LIMIT_SLIPPAGE = 0.001   # 0.1 %


# ─────────────────────────────────────────────────────────────────────────────
# Enums & dataclasses
# ─────────────────────────────────────────────────────────────────────────────

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

    trade_id: str                           # Unique ID linking signal → fill
    order_id: str                           # Broker-assigned order ID
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
    cancel_timer: Optional[threading.Timer] = None   # fires if limit order stale
    retry_at_market: bool = True
    broker_response: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# OrderExecutor
# ─────────────────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Places and manages orders through the Alpaca brokerage.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client.
    use_limit_orders : bool
        Default True — submit limit orders at current price ± 0.1 %.
    limit_slippage_pct : float
        Limit price offset as a fraction of current price (default 0.001).
    stale_order_seconds : int
        Cancel unfilled limit orders after this many seconds (default 30).
    retry_at_market : bool
        If True, re-submit as a market order after a stale limit cancel.
    on_fill : Callable | None
        Optional callback invoked with the Order when a fill is confirmed.
    """

    def __init__(
        self,
        client: AlpacaClient,
        use_limit_orders: bool = True,
        limit_slippage_pct: float = _LIMIT_SLIPPAGE,
        stale_order_seconds: int = _LIMIT_CANCEL_SECONDS,
        retry_at_market: bool = True,
        on_fill: Optional[Callable[[Order], None]] = None,
    ) -> None:
        self.client = client
        self.use_limit_orders = use_limit_orders
        self.limit_slippage_pct = limit_slippage_pct
        self.stale_order_seconds = stale_order_seconds
        self.retry_at_market = retry_at_market
        self.on_fill = on_fill

        # order_id → Order
        self._open_orders: dict[str, Order] = {}
        # trade_id → Order (for signal linkage)
        self._trade_log: dict[str, Order] = {}
        # symbols with any open or pending order
        self._pending_symbols: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def submit_order(
        self,
        signal,                         # core.regime_strategies.Signal or TradingSignal
        current_price: float,
        qty: float,
        trade_id: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Submit a limit order for `signal`.

        Parameters
        ----------
        signal : Signal | TradingSignal
        current_price : float
            Latest market price used to set limit price.
        qty : float
            Share quantity (already sized by RiskManager).
        trade_id : str | None
            Caller-supplied linkage ID.  Auto-generated if None.

        Returns
        -------
        Order | None
            Submitted order, or None if skipped (duplicate, zero qty, etc.)
        """
        symbol = signal.symbol
        direction = str(getattr(signal, "direction", getattr(signal, "action", "LONG"))).upper()
        side = OrderSide.BUY if "BUY" in direction or "LONG" in direction else OrderSide.SELL

        if qty <= 0:
            logger.debug("submit_order: skipped zero qty for %s", symbol)
            return None

        with self._lock:
            if self._is_duplicate(symbol):
                logger.info("submit_order: duplicate suppressed for %s", symbol)
                return None

        tid = trade_id or str(uuid.uuid4())

        if self.use_limit_orders:
            slippage = self.limit_slippage_pct
            lim_price = (
                round(current_price * (1 + slippage), 2)
                if side == OrderSide.BUY
                else round(current_price * (1 - slippage), 2)
            )
            order_type = OrderType.LIMIT
        else:
            lim_price = None
            order_type = OrderType.MARKET

        try:
            broker_resp = self._submit_to_broker(
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=qty,
                limit_price=lim_price,
                stop_price=None,
                client_order_id=tid,
            )
        except Exception as exc:
            logger.error("submit_order failed for %s: %s", symbol, exc)
            return None

        order = Order(
            trade_id=tid,
            order_id=broker_resp.get("id", tid),
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=lim_price,
            stop_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now(tz=timezone.utc),
            retry_at_market=self.retry_at_market,
            broker_response=broker_resp,
        )

        with self._lock:
            self._open_orders[order.order_id] = order
            self._trade_log[tid] = order
            self._pending_symbols.add(symbol)

        if order_type == OrderType.LIMIT:
            self._schedule_cancel(order)

        logger.info(
            "Order submitted: %s %s %s qty=%.2f lim=%.2f trade_id=%s",
            side.value.upper(), qty, symbol, qty, lim_price or 0, tid,
        )
        return order

    def submit_bracket_order(
        self,
        signal,
        current_price: float,
        qty: float,
        trade_id: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Submit an OCO bracket order: entry + stop-loss + take-profit.

        Requires signal.stop_loss and signal.take_profit to be set.

        Parameters
        ----------
        signal : Signal
        current_price : float
        qty : float
        trade_id : str | None

        Returns
        -------
        Order | None
        """
        stop_loss = getattr(signal, "stop_loss", None)
        take_profit = getattr(signal, "take_profit", None)

        if stop_loss is None or take_profit is None:
            logger.warning(
                "submit_bracket_order: %s missing stop_loss or take_profit — "
                "falling back to plain order",
                signal.symbol,
            )
            return self.submit_order(signal, current_price, qty, trade_id)

        symbol = signal.symbol
        direction = str(getattr(signal, "direction", "LONG")).upper()
        side = OrderSide.BUY if "LONG" in direction or "BUY" in direction else OrderSide.SELL

        if qty <= 0:
            return None

        with self._lock:
            if self._is_duplicate(symbol):
                logger.info("submit_bracket_order: duplicate suppressed for %s", symbol)
                return None

        tid = trade_id or str(uuid.uuid4())
        lim_price = (
            round(current_price * (1 + self.limit_slippage_pct), 2)
            if side == OrderSide.BUY
            else round(current_price * (1 - self.limit_slippage_pct), 2)
        )

        try:
            broker_resp = self._submit_bracket_to_broker(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=lim_price,
                stop_loss_price=float(stop_loss),
                take_profit_price=float(take_profit),
                client_order_id=tid,
            )
        except Exception as exc:
            logger.error("submit_bracket_order failed for %s: %s", symbol, exc)
            return None

        order = Order(
            trade_id=tid,
            order_id=broker_resp.get("id", tid),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            qty=qty,
            limit_price=lim_price,
            stop_price=float(stop_loss),
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now(tz=timezone.utc),
            retry_at_market=False,  # brackets don't retry at market
            broker_response=broker_resp,
        )

        with self._lock:
            self._open_orders[order.order_id] = order
            self._trade_log[tid] = order
            self._pending_symbols.add(symbol)

        logger.info(
            "Bracket order submitted: %s %s qty=%.2f entry=%.2f stop=%.2f tp=%.2f trade_id=%s",
            side.value.upper(), symbol, qty, lim_price,
            float(stop_loss), float(take_profit), tid,
        )
        return order

    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """
        Tighten the stop-loss on an open position.

        This method ONLY tightens stops — it will NEVER widen them.
        A tighter stop means a higher price for a long, lower for a short.

        Parameters
        ----------
        symbol : str
        new_stop : float

        Returns
        -------
        bool
            True if the stop was successfully modified.
        """
        # Find an open order or use the positions API
        # We need to check the current stop before tightening
        positions = self.client.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)
        if pos is None:
            logger.warning("modify_stop: no open position for %s", symbol)
            return False

        # For a long position, new_stop must be HIGHER than current stop
        # For a short position, new_stop must be LOWER
        # We'll call Alpaca's replace_order on the associated stop order
        side = pos.get("side", "long")

        # Find the stop order for this symbol
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
        except ImportError:
            logger.error("alpaca-py not installed")
            return False

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = self.client._retry_request(
            self.client._trading_client.get_orders, req
        )

        stop_order = None
        for o in open_orders:
            if o.symbol == symbol and str(o.order_type).lower() == "stop":
                stop_order = o
                break

        if stop_order is None:
            logger.warning(
                "modify_stop: no open stop order found for %s — "
                "cannot modify via order replacement",
                symbol,
            )
            return False

        current_stop = float(stop_order.stop_price or 0)
        if current_stop == 0:
            return False

        # Tighten check
        is_long = side == "long"
        if is_long and new_stop <= current_stop:
            logger.info(
                "modify_stop: rejected for %s — new_stop %.4f would widen "
                "(current=%.4f, long position)",
                symbol, new_stop, current_stop,
            )
            return False
        if not is_long and new_stop >= current_stop:
            logger.info(
                "modify_stop: rejected for %s — new_stop %.4f would widen "
                "(current=%.4f, short position)",
                symbol, new_stop, current_stop,
            )
            return False

        try:
            from alpaca.trading.requests import ReplaceOrderRequest
        except ImportError:
            logger.error("alpaca-py not installed")
            return False

        replace_req = ReplaceOrderRequest(stop_price=str(new_stop))
        self.client._retry_request(
            self.client._trading_client.replace_order_by_id,
            str(stop_order.id),
            replace_req,
        )
        logger.info(
            "modify_stop: %s stop tightened %.4f → %.4f",
            symbol, current_stop, new_stop,
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by broker order ID.

        Returns
        -------
        bool
            True if the cancel request was accepted.
        """
        try:
            self.client._retry_request(
                self.client._trading_client.cancel_order_by_id, order_id
            )
        except Exception as exc:
            logger.error("cancel_order %s failed: %s", order_id, exc)
            return False

        with self._lock:
            order = self._open_orders.pop(order_id, None)
            if order:
                order.status = OrderStatus.CANCELLED
                self._pending_symbols.discard(order.symbol)
                if order.cancel_timer:
                    order.cancel_timer.cancel()

        logger.info("Order cancelled: %s", order_id)
        return True

    def cancel_all(self) -> list[str]:
        """Cancel all open orders.  Returns list of cancelled order IDs."""
        with self._lock:
            ids = list(self._open_orders.keys())

        cancelled: list[str] = []
        for oid in ids:
            if self.cancel_order(oid):
                cancelled.append(oid)
        return cancelled

    def cancel_stale_orders(self) -> list[str]:
        """
        Cancel any open limit orders older than stale_order_seconds.

        Returns
        -------
        list[str]
            Cancelled order IDs.
        """
        now = datetime.now(tz=timezone.utc)
        stale: list[str] = []

        with self._lock:
            snapshot = list(self._open_orders.values())

        for order in snapshot:
            if order.order_type != OrderType.LIMIT:
                continue
            age = (now - order.submitted_at).total_seconds()
            if age >= self.stale_order_seconds:
                stale.append(order.order_id)

        cancelled: list[str] = []
        for oid in stale:
            if self.cancel_order(oid):
                cancelled.append(oid)
                if self.retry_at_market:
                    self._retry_as_market(oid)
        return cancelled

    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Market-sell (or cover) the entire open position for `symbol`.

        Returns
        -------
        Order | None
        """
        positions = self.client.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)
        if pos is None:
            logger.info("close_position: no open position for %s", symbol)
            return None

        qty = abs(float(pos["qty"]))
        side = OrderSide.SELL if pos["side"] == "long" else OrderSide.BUY
        tid = str(uuid.uuid4())

        try:
            broker_resp = self._submit_to_broker(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                qty=qty,
                limit_price=None,
                stop_price=None,
                client_order_id=tid,
            )
        except Exception as exc:
            logger.error("close_position %s failed: %s", symbol, exc)
            return None

        order = Order(
            trade_id=tid,
            order_id=broker_resp.get("id", tid),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
            limit_price=None,
            stop_price=None,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now(tz=timezone.utc),
            broker_response=broker_resp,
        )
        with self._lock:
            self._open_orders[order.order_id] = order
            self._trade_log[tid] = order

        logger.info("close_position: market %s %s qty=%.2f", side.value, symbol, qty)
        return order

    def close_all_positions(self) -> list[Order]:
        """Close all open positions with market orders."""
        positions = self.client.get_positions()
        orders: list[Order] = []
        for pos in positions:
            order = self.close_position(pos["symbol"])
            if order:
                orders.append(order)
        return orders

    def sync_order_statuses(self) -> None:
        """
        Query Alpaca for the current status of all tracked open orders
        and update internal state.  Call at the start of each trading loop.
        """
        with self._lock:
            ids = list(self._open_orders.keys())

        for oid in ids:
            try:
                raw = self.client._retry_request(
                    self.client._trading_client.get_order_by_id, oid
                )
            except Exception as exc:
                logger.warning("sync_order_statuses: error fetching %s: %s", oid, exc)
                continue

            status_str = str(raw.status).lower()
            with self._lock:
                order = self._open_orders.get(oid)
                if order is None:
                    continue
                order.filled_qty = float(raw.filled_qty or 0)
                order.filled_avg_price = float(raw.filled_avg_price or 0) or None
                if "fill" in status_str:
                    order.status = OrderStatus.FILLED
                    order.filled_at = raw.filled_at
                    self._open_orders.pop(oid, None)
                    self._pending_symbols.discard(order.symbol)
                    if order.cancel_timer:
                        order.cancel_timer.cancel()
                    if self.on_fill:
                        self.on_fill(order)
                    logger.info(
                        "Fill confirmed: %s %s qty=%.2f @ %.4f trade_id=%s",
                        order.side.value, order.symbol,
                        order.filled_qty, order.filled_avg_price or 0,
                        order.trade_id,
                    )
                elif status_str in ("cancelled", "canceled", "expired", "rejected"):
                    order.status = OrderStatus.CANCELLED
                    self._open_orders.pop(oid, None)
                    self._pending_symbols.discard(order.symbol)
                elif "partial" in status_str:
                    order.status = OrderStatus.PARTIAL

    def get_trade(self, trade_id: str) -> Optional[Order]:
        """Look up a historical order by trade_id."""
        return self._trade_log.get(trade_id)

    # ------------------------------------------------------------------
    # Backward-compat (used by signal_generator)
    # ------------------------------------------------------------------

    def execute_signal(self, signal, equity: float) -> Optional[Order]:
        """
        Convert a TradingSignal into a submitted order (legacy interface).

        Computes share quantity from signal.target_weight × equity / price,
        then delegates to submit_order().
        """
        action_raw = getattr(signal, "action", "hold")
        # str-Enum members expose .value; plain strings pass through unchanged
        action = (action_raw.value if hasattr(action_raw, "value") else str(action_raw)).upper()
        if action in ("HOLD", "CLOSE"):
            if action == "CLOSE":
                return self.close_position(signal.symbol)
            return None

        latest_bar = self.client.get_latest_bar(signal.symbol)
        if latest_bar is None:
            logger.warning("execute_signal: no price for %s", signal.symbol)
            return None

        price = latest_bar["close"]
        weight = getattr(signal, "target_weight", 0.0)
        qty = _compute_qty(weight, price, equity)
        return self.submit_order(signal, price, qty)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_to_broker(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        limit_price: Optional[float],
        stop_price: Optional[float],
        client_order_id: str,
    ) -> dict[str, Any]:
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaSide,
                TimeInForce,
            )
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL
        tif = TimeInForce.DAY

        if order_type == OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=tif,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.LIMIT:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=tif,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.STOP:
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=tif,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
        else:
            req = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=tif,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )

        resp = self.client._retry_request(
            self.client._trading_client.submit_order, req
        )
        return {"id": str(resp.id), "status": str(resp.status)}

    def _submit_bracket_to_broker(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        try:
            from alpaca.trading.requests import (
                LimitOrderRequest,
                StopLossRequest,
                TakeProfitRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaSide,
                TimeInForce,
                OrderClass,
            )
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
            limit_price=limit_price,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            client_order_id=client_order_id,
        )
        resp = self.client._retry_request(
            self.client._trading_client.submit_order, req
        )
        return {"id": str(resp.id), "status": str(resp.status)}

    def _schedule_cancel(self, order: Order) -> None:
        """Schedule automatic cancellation after stale_order_seconds."""
        def _cancel():
            logger.info(
                "Stale limit order — cancelling %s after %ds",
                order.order_id, self.stale_order_seconds,
            )
            self.cancel_order(order.order_id)
            if order.retry_at_market and self.retry_at_market:
                self._retry_as_market(order.order_id)

        timer = threading.Timer(self.stale_order_seconds, _cancel)
        timer.daemon = True
        timer.start()
        order.cancel_timer = timer

    def _retry_as_market(self, original_order_id: str) -> None:
        """Re-submit a previously-cancelled limit order as a market order."""
        order = self._trade_log.get(original_order_id) or self._open_orders.get(original_order_id)
        if order is None:
            return
        logger.info("Retrying %s %s as market order", order.symbol, order.side.value)
        try:
            broker_resp = self._submit_to_broker(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                qty=order.qty - order.filled_qty,
                limit_price=None,
                stop_price=None,
                client_order_id=str(uuid.uuid4()),
            )
            retry_order = Order(
                trade_id=order.trade_id + "_retry",
                order_id=broker_resp.get("id", ""),
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                qty=order.qty - order.filled_qty,
                limit_price=None,
                stop_price=None,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(tz=timezone.utc),
                broker_response=broker_resp,
            )
            with self._lock:
                self._open_orders[retry_order.order_id] = retry_order
                self._pending_symbols.add(retry_order.symbol)
        except Exception as exc:
            logger.error("Market retry for %s failed: %s", order.symbol, exc)

    def _is_duplicate(self, symbol: str) -> bool:
        """Return True if there is already an open or pending order for symbol."""
        return symbol in self._pending_symbols


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_qty(weight: float, price: float, equity: float) -> float:
    """Convert target portfolio weight → share quantity (floor to 2 dp)."""
    if price <= 0 or equity <= 0:
        return 0.0
    notional = weight * equity
    return round(notional / price, 2)
