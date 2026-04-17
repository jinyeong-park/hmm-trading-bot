"""
dashboard.py — Terminal-based live trading dashboard.

Responsibilities:
  - Render a Rich Live layout showing positions, P&L, regime state, and recent signals.
  - Refresh on a configurable interval without blocking the main trading loop.
  - Display drawdown gauges and risk-limit status.
  - Support graceful start/stop from the main process.
"""

from __future__ import annotations

import threading
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live

from broker.position_tracker import PortfolioSnapshot
from core.hmm_engine import RegimeResult
from core.signal_generator import TradingSignal


class TradingDashboard:
    """
    Terminal dashboard rendered via Rich Live.

    Parameters
    ----------
    refresh_seconds : int
        How often to redraw the dashboard.
    console : rich.Console | None
        Optional pre-configured console (useful for testing).
    """

    def __init__(
        self,
        refresh_seconds: int = 5,
        console: Optional[Console] = None,
    ) -> None:
        self.refresh_seconds = refresh_seconds
        self.console = console or Console()

        self._live: Optional[Live] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Shared state updated by the trading loop
        self._snapshot: Optional[PortfolioSnapshot] = None
        self._regime_results: dict[str, RegimeResult] = {}
        self._recent_signals: list[TradingSignal] = []
        self._alerts: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the dashboard refresh loop in a daemon thread.
        Non-blocking — returns immediately.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the refresh loop and restore the terminal."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State updates (called from the trading loop thread)
    # ------------------------------------------------------------------

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Replace the cached PortfolioSnapshot used for rendering."""
        raise NotImplementedError

    def update_regime(self, symbol: str, result: RegimeResult) -> None:
        """Update the regime state for a specific symbol."""
        raise NotImplementedError

    def update_signals(self, signals: list[TradingSignal]) -> None:
        """Replace the recent signals list shown in the signal panel."""
        raise NotImplementedError

    def add_alert(self, message: str) -> None:
        """Append a timestamped alert string to the alert panel."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rendering (called from the refresh thread)
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        """
        Construct and return the full Rich Layout for one render frame.

        Layout structure:
          ┌─────────────────────────────────────┐
          │  header (bot name, clock, market)   │
          ├──────────────┬──────────────────────┤
          │  positions   │  regime / signals    │
          ├──────────────┴──────────────────────┤
          │  drawdown / risk gauges             │
          ├─────────────────────────────────────┤
          │  recent alerts                      │
          └─────────────────────────────────────┘

        Returns
        -------
        Layout
        """
        raise NotImplementedError

    def _render_header(self) -> "RenderableType":  # noqa: F821
        """Return the header panel (bot name, UTC clock, market open/closed)."""
        raise NotImplementedError

    def _render_positions(self) -> "RenderableType":  # noqa: F821
        """Return a Rich Table of open positions with P&L columns."""
        raise NotImplementedError

    def _render_regime(self) -> "RenderableType":  # noqa: F821
        """Return a panel showing per-symbol regime and confidence."""
        raise NotImplementedError

    def _render_risk_gauges(self) -> "RenderableType":  # noqa: F821
        """Return drawdown bars and risk-limit status indicators."""
        raise NotImplementedError

    def _render_alerts(self) -> "RenderableType":  # noqa: F821
        """Return the last N alert lines."""
        raise NotImplementedError
