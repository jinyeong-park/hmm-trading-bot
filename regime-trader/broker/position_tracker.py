"""
position_tracker.py — Track open positions and real-time P&L.

Responsibilities:
  - Sync open positions from Alpaca at startup and after each fill.
  - Compute unrealised P&L, realised P&L, and portfolio-level statistics.
  - Maintain a local position ledger to avoid excessive API calls.
  - Detect and reconcile discrepancies between local state and broker state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from broker.alpaca_client import AlpacaClient


@dataclass
class Position:
    """Snapshot of a single open position."""

    symbol: str
    qty: float                          # Positive = long, negative = short
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealised_pnl: float
    unrealised_pnl_pct: float
    side: str                           # 'long' | 'short'
    opened_at: Optional[datetime] = None
    asset_class: str = "us_equity"


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


class PositionTracker:
    """
    Maintains a real-time local view of open positions and portfolio P&L.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client used for sync calls.
    sync_interval_seconds : int
        How often (in seconds) to force a full broker reconciliation.
    """

    def __init__(
        self,
        client: AlpacaClient,
        sync_interval_seconds: int = 60,
    ) -> None:
        self.client = client
        self.sync_interval_seconds = sync_interval_seconds

        self._positions: dict[str, Position] = {}
        self._last_sync: Optional[datetime] = None
        self._realised_pnl_today: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> PortfolioSnapshot:
        """
        Pull the latest positions and account data from Alpaca and update
        the internal ledger.

        Returns
        -------
        PortfolioSnapshot
        """
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Return the current position for `symbol`, or None if flat.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        Position | None
        """
        raise NotImplementedError

    def get_all_positions(self) -> dict[str, Position]:
        """Return a copy of the current position ledger."""
        raise NotImplementedError

    def get_snapshot(self) -> PortfolioSnapshot:
        """
        Return the most recent PortfolioSnapshot from the last sync.
        Does NOT trigger a fresh API call — call sync() first if freshness matters.
        """
        raise NotImplementedError

    def get_current_weights(self, equity: float) -> dict[str, float]:
        """
        Compute each position's weight as a fraction of portfolio equity.

        Parameters
        ----------
        equity : float

        Returns
        -------
        dict[str, float]
            symbol → weight
        """
        raise NotImplementedError

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float, side: str) -> None:
        """
        Update the local ledger immediately upon receiving a fill event,
        without waiting for the next full sync.

        Parameters
        ----------
        order_id : str
        symbol : str
        qty : float
        price : float
        side : str
            'buy' | 'sell'
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reconcile(
        self,
        broker_positions: list[dict],
        local_positions: dict[str, Position],
    ) -> dict[str, Position]:
        """
        Reconcile broker-reported positions against local ledger.
        Logs any discrepancies detected.

        Parameters
        ----------
        broker_positions : list[dict]
            Raw position dicts from Alpaca API.
        local_positions : dict[str, Position]

        Returns
        -------
        dict[str, Position]
            Reconciled position ledger.
        """
        raise NotImplementedError

    def _parse_broker_position(self, raw: dict) -> Position:
        """
        Convert a raw Alpaca position dict into a Position dataclass.

        Parameters
        ----------
        raw : dict

        Returns
        -------
        Position
        """
        raise NotImplementedError
