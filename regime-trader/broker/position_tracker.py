"""
position_tracker.py — Real-time position ledger with WebSocket fill integration.

Responsibilities:
  - Subscribe to Alpaca trade-update WebSocket for instant fill notifications.
  - Update PortfolioState and CircuitBreaker on every fill without waiting for sync.
  - Track per-position: entry time/price, current price, unrealised P&L, stop level,
    holding period, and the regime label at entry vs current.
  - Reconcile local ledger with Alpaca broker state on startup and periodically.
  - Expose a snapshot suitable for passing directly to RiskManager.validate_signal().
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.risk_manager import CircuitBreaker, PortfolioState
from core.risk_manager import Position as RiskPosition

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Full per-position state tracked by the position tracker."""

    symbol: str
    qty: float                              # Positive = long, negative = short
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealised_pnl: float
    unrealised_pnl_pct: float
    side: str                               # 'long' | 'short'
    stop_price: float = 0.0                 # Current hard stop level
    take_profit_price: float = 0.0          # Current take-profit level
    opened_at: Optional[datetime] = None
    asset_class: str = "us_equity"
    sector: str = "UNKNOWN"
    regime_at_entry: str = "UNKNOWN"        # Regime when position was opened
    regime_current: str = "UNKNOWN"         # Most recent regime label
    trade_id: str = ""                      # Links to OrderExecutor trade_id

    @property
    def holding_period_seconds(self) -> float:
        """Seconds since position was opened (0 if opened_at is None)."""
        if self.opened_at is None:
            return 0.0
        return (datetime.now(tz=timezone.utc) - self.opened_at).total_seconds()

    @property
    def holding_period_days(self) -> float:
        return self.holding_period_seconds / 86_400


@dataclass
class PortfolioSnapshot:
    """Aggregated portfolio-level statistics."""

    equity: float
    cash: float
    buying_power: float
    gross_exposure: float
    net_exposure: float
    long_market_value: float
    short_market_value: float
    unrealised_pnl: float
    realised_pnl_today: float
    positions: dict[str, Position] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# PositionTracker
# ─────────────────────────────────────────────────────────────────────────────

class PositionTracker:
    """
    Maintains a real-time local view of open positions and portfolio P&L.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client.
    circuit_breaker : CircuitBreaker | None
        If provided, `circuit_breaker.update()` is called on every fill.
    sync_interval_seconds : int
        Minimum time between full broker reconciliation calls.
    """

    def __init__(
        self,
        client: AlpacaClient,
        circuit_breaker: Optional[CircuitBreaker] = None,
        sync_interval_seconds: int = 60,
    ) -> None:
        self.client = client
        self.circuit_breaker = circuit_breaker
        self.sync_interval_seconds = sync_interval_seconds

        self._positions: dict[str, Position] = {}
        self._snapshot: Optional[PortfolioSnapshot] = None
        self._last_sync: Optional[datetime] = None
        self._realised_pnl_today: float = 0.0
        self._lock = threading.RLock()
        self._current_regime: str = "UNKNOWN"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Perform initial sync and subscribe to trade-update WebSocket.
        Call once after AlpacaClient.connect().
        """
        self.sync()
        self.client.subscribe_trade_updates(self._on_trade_update)
        logger.info("PositionTracker started: %d positions loaded", len(self._positions))

    def set_current_regime(self, regime_label: str) -> None:
        """Update the current regime label used for new position annotations."""
        self._current_regime = regime_label
        with self._lock:
            for pos in self._positions.values():
                pos.regime_current = regime_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> PortfolioSnapshot:
        """
        Pull the latest positions and account data from Alpaca and reconcile
        with the local ledger.

        Returns
        -------
        PortfolioSnapshot
        """
        acc = self.client.get_account()
        broker_positions = self.client.get_positions()

        with self._lock:
            self._positions = self._reconcile(broker_positions, self._positions)
            snapshot = self._build_snapshot(acc)
            self._snapshot = snapshot
            self._last_sync = datetime.now(tz=timezone.utc)

        if self.circuit_breaker is not None:
            self.circuit_breaker.update(snapshot.equity, self._last_sync)

        logger.debug(
            "PositionTracker.sync: equity=%.2f  n_pos=%d",
            snapshot.equity, len(self._positions),
        )
        return snapshot

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the current position for `symbol`, or None if flat."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Return a copy of the current position ledger."""
        with self._lock:
            return dict(self._positions)

    def get_snapshot(self) -> PortfolioSnapshot:
        """
        Return the most recent PortfolioSnapshot without triggering an API call.
        Call sync() first if freshness matters.
        """
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(
                    "No snapshot available — call sync() or start() first"
                )
            return self._snapshot

    def get_current_weights(self, equity: float) -> dict[str, float]:
        """Compute each position's weight as a fraction of portfolio equity."""
        if equity <= 0:
            return {}
        with self._lock:
            return {
                sym: pos.market_value / equity
                for sym, pos in self._positions.items()
            }

    def get_portfolio_state(
        self,
        flicker_rate: float = 0.0,
        price_history: Optional[dict] = None,
    ) -> PortfolioState:
        """
        Build a RiskManager-compatible PortfolioState from current snapshot.

        Parameters
        ----------
        flicker_rate : float
            Current HMM flicker rate (transitions / flicker_window).
        price_history : dict | None
            symbol → pd.Series of closes for correlation checks.
        """
        snap = self.get_snapshot()
        risk_positions: dict[str, RiskPosition] = {}
        with self._lock:
            for sym, pos in self._positions.items():
                risk_positions[sym] = RiskPosition(
                    symbol=sym,
                    shares=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=pos.current_price,
                    stop_price=pos.stop_price,
                    sector=pos.sector,
                )
        return PortfolioState(
            equity=snap.equity,
            cash=snap.cash,
            buying_power=snap.buying_power,
            positions=risk_positions,
            daily_pnl=snap.realised_pnl_today,
            weekly_pnl=0.0,             # updated externally via update_equity
            peak_equity=snap.equity,    # circuit_breaker tracks this properly
            current_drawdown=0.0,
            circuit_breaker_status={},
            flicker_rate=flicker_rate,
            price_history=price_history or {},
            timestamp=snap.timestamp,
        )

    def on_fill(
        self,
        order_id: str,
        symbol: str,
        qty: float,
        price: float,
        side: str,
        trade_id: str = "",
        stop_price: float = 0.0,
        take_profit_price: float = 0.0,
    ) -> None:
        """
        Update the local ledger immediately on a fill event without waiting for sync.

        Parameters
        ----------
        order_id : str
        symbol : str
        qty : float
            Shares filled (always positive).
        price : float
            Fill price.
        side : str
            'buy' | 'sell'
        trade_id : str
        stop_price : float
        take_profit_price : float
        """
        is_buy = side.lower() == "buy"
        now = datetime.now(tz=timezone.utc)

        with self._lock:
            existing = self._positions.get(symbol)

            if is_buy:
                if existing is None:
                    # New long position
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        qty=qty,
                        avg_entry_price=price,
                        current_price=price,
                        market_value=qty * price,
                        unrealised_pnl=0.0,
                        unrealised_pnl_pct=0.0,
                        side="long",
                        opened_at=now,
                        stop_price=stop_price,
                        take_profit_price=take_profit_price,
                        regime_at_entry=self._current_regime,
                        regime_current=self._current_regime,
                        trade_id=trade_id,
                    )
                    logger.info("New position: BUY %s qty=%.2f @ %.4f", symbol, qty, price)
                else:
                    # Average up
                    total_qty = existing.qty + qty
                    existing.avg_entry_price = (
                        (existing.avg_entry_price * existing.qty + price * qty) / total_qty
                    )
                    existing.qty = total_qty
                    existing.market_value = total_qty * price
                    logger.info("Added to position: %s qty now %.2f", symbol, total_qty)

            else:  # sell
                if existing is None:
                    logger.warning(
                        "on_fill: received SELL fill for %s but no local position tracked",
                        symbol,
                    )
                    return
                closed_qty = min(qty, abs(existing.qty))
                realised = closed_qty * (price - existing.avg_entry_price)
                self._realised_pnl_today += realised

                remaining = existing.qty - closed_qty
                if remaining <= 1e-4:
                    del self._positions[symbol]
                    logger.info(
                        "Position closed: %s qty=%.2f realised_pnl=%.2f",
                        symbol, closed_qty, realised,
                    )
                else:
                    existing.qty = remaining
                    existing.market_value = remaining * price
                    logger.info(
                        "Position reduced: %s qty now %.2f", symbol, remaining
                    )

        # Update circuit breaker with new equity estimate
        if self.circuit_breaker is not None and self._snapshot is not None:
            with self._lock:
                total_mv = sum(p.market_value for p in self._positions.values())
                new_equity = self._snapshot.cash + total_mv
            self.circuit_breaker.update(new_equity, now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_trade_update(self, update: dict) -> None:
        """
        WebSocket callback for Alpaca trade-update events.
        Dispatches fills to on_fill(); logs all other events.
        """
        event = update.get("event", "")
        symbol = update.get("symbol", "")
        logger.debug("Trade update [%s]: %s", event, symbol)

        if event in ("fill", "partial_fill"):
            qty = float(update.get("filled_qty", 0))
            price = float(update.get("filled_avg_price", 0))
            side = update.get("side", "")
            order_id = update.get("order_id", "")
            trade_id = update.get("client_order_id", "")

            if qty > 0 and price > 0:
                self.on_fill(
                    order_id=order_id,
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    side=side,
                    trade_id=trade_id,
                )
            else:
                logger.warning(
                    "Fill event with zero qty or price: %s %s", symbol, update
                )

        elif event == "canceled":
            logger.info("Order cancelled: %s %s", symbol, update.get("order_id"))
        elif event == "rejected":
            logger.warning("Order rejected: %s reason=%s", symbol, update)

    def _reconcile(
        self,
        broker_positions: list[dict],
        local_positions: dict[str, Position],
    ) -> dict[str, Position]:
        """
        Reconcile broker-reported positions against the local ledger.
        Logs discrepancies.  Broker state is authoritative.
        """
        broker_by_sym = {p["symbol"]: p for p in broker_positions}
        reconciled: dict[str, Position] = {}

        for sym, raw in broker_by_sym.items():
            broker_qty = float(raw["qty"])
            if sym in local_positions:
                local = local_positions[sym]
                if abs(local.qty - broker_qty) > 0.01:
                    logger.warning(
                        "Reconciliation mismatch for %s: local=%.4f broker=%.4f — "
                        "adopting broker value",
                        sym, local.qty, broker_qty,
                    )
                pos = self._parse_broker_position(raw)
                # Preserve metadata that only we know
                pos.opened_at = local.opened_at
                pos.stop_price = local.stop_price
                pos.take_profit_price = local.take_profit_price
                pos.regime_at_entry = local.regime_at_entry
                pos.regime_current = local.regime_current
                pos.trade_id = local.trade_id
                pos.sector = local.sector
            else:
                pos = self._parse_broker_position(raw)
                logger.info(
                    "Discovered untracked broker position: %s qty=%.4f — adding",
                    sym, broker_qty,
                )
            reconciled[sym] = pos

        # Positions in local but not in broker have been closed externally
        for sym in local_positions:
            if sym not in broker_by_sym:
                logger.warning(
                    "Local position %s not found at broker — assuming closed",
                    sym,
                )

        return reconciled

    def _parse_broker_position(self, raw: dict) -> Position:
        """Convert a raw Alpaca position dict into a Position dataclass."""
        qty = float(raw.get("qty", 0))
        entry = float(raw.get("avg_entry_price", 0))
        current = float(raw.get("current_price", entry))
        mv = float(raw.get("market_value", qty * current))
        unreal = float(raw.get("unrealised_pl", 0))
        unreal_pct = float(raw.get("unrealised_plpc", 0))
        return Position(
            symbol=raw["symbol"],
            qty=qty,
            avg_entry_price=entry,
            current_price=current,
            market_value=mv,
            unrealised_pnl=unreal,
            unrealised_pnl_pct=unreal_pct,
            side=raw.get("side", "long"),
            asset_class=raw.get("asset_class", "us_equity"),
            regime_current=self._current_regime,
        )

    def _build_snapshot(self, acc: dict) -> PortfolioSnapshot:
        """Build PortfolioSnapshot from account dict and current positions."""
        equity = acc.get("equity", 0.0)
        cash = acc.get("cash", 0.0)
        bp = acc.get("buying_power", 0.0)

        long_mv = sum(
            p.market_value for p in self._positions.values() if p.side == "long"
        )
        short_mv = sum(
            abs(p.market_value) for p in self._positions.values() if p.side == "short"
        )
        total_unreal = sum(p.unrealised_pnl for p in self._positions.values())
        gross_exp = (long_mv + short_mv) / equity if equity > 0 else 0.0
        net_exp = (long_mv - short_mv) / equity if equity > 0 else 0.0

        return PortfolioSnapshot(
            equity=float(equity),
            cash=float(cash),
            buying_power=float(bp),
            gross_exposure=round(gross_exp, 6),
            net_exposure=round(net_exp, 6),
            long_market_value=long_mv,
            short_market_value=short_mv,
            unrealised_pnl=total_unreal,
            realised_pnl_today=self._realised_pnl_today,
            positions=dict(self._positions),
            timestamp=datetime.now(tz=timezone.utc),
        )
