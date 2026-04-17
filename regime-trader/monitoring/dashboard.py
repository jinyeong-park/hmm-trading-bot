"""
dashboard.py — Terminal live trading dashboard (Rich).

Layout (refreshes every 5 seconds):

  ┌─ regime-trader ── 14:32:07 UTC ── PAPER ─────────────────────┐
  ├─ REGIME ─────────────────────────────────────────────────────┤
  │  BULL (72%)  │  Stability: 14 bars  │  Flicker: 1/20  ✅     │
  ├─ PORTFOLIO ──────────────────────────────────────────────────┤
  │  Equity: $105,230  │  Daily: +$340 (+0.32%)                  │
  │  Allocation: 95%   │  Leverage: 1.25×                        │
  ├─ POSITIONS ──────────────────────────────────────────────────┤
  │  SYMBOL  SIDE  PRICE    P&L      STOP     HOLDING            │
  │  SPY     LONG  $520.30  +1.2%    $508.00  3h 12m             │
  ├─ RECENT SIGNALS ─────────────────────────────────────────────┤
  │  14:30  SPY  Rebalance 60%→95%  Low vol                      │
  ├─ RISK STATUS ────────────────────────────────────────────────┤
  │  Daily DD:    0.3% / 3.0%  ████░░░░░░  ✅                    │
  │  Weekly DD:   0.5% / 7.0%  █░░░░░░░░░  ✅                    │
  │  Peak DD:     1.2% / 10.0% ██░░░░░░░░  ✅                    │
  ├─ SYSTEM ─────────────────────────────────────────────────────┤
  │  Data: ✅  │  API: ✅ 23ms  │  HMM: 2d ago  │  PAPER         │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID
from rich.table import Table
from rich.text import Text

from broker.position_tracker import PortfolioSnapshot

# ─────────────────────────────────────────────────────────────────────────────
# Colour constants
# ─────────────────────────────────────────────────────────────────────────────
_C_OK      = "bold green"
_C_WARN    = "bold yellow"
_C_CRIT    = "bold red"
_C_REGIME  = {
    "BULL": "bold green", "STRONG_BULL": "bold green",
    "WEAK_BULL": "green",
    "NEUTRAL": "bold white",
    "BEAR": "red", "STRONG_BEAR": "bold red",
    "WEAK_BEAR": "dark_orange",
    "CRASH": "bold red on white",
    "EUPHORIA": "bold magenta",
    "UNKNOWN": "dim",
    "LOW_VOL": "bold green", "MID_VOL": "bold yellow", "HIGH_VOL": "bold red",
}
_MAX_SIGNALS = 6
_MAX_ALERTS  = 8


# ─────────────────────────────────────────────────────────────────────────────
# DashboardState  (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class DashboardState:
    """All mutable data that feeds the dashboard render."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.snapshot: Optional[PortfolioSnapshot] = None

        # Regime per primary symbol
        self.regime_label: str = "UNKNOWN"
        self.regime_probability: float = 0.0
        self.regime_consecutive_bars: int = 0
        self.flicker_count: int = 0
        self.flicker_window: int = 20
        self.is_confirmed: bool = False

        # Portfolio-level metrics
        self.allocation_pct: float = 0.0
        self.leverage: float = 1.0
        self.daily_open_equity: float = 0.0

        # Drawdown gauges
        self.daily_dd: float = 0.0
        self.weekly_dd: float = 0.0
        self.peak_dd: float = 0.0
        self.daily_dd_halt: float = 0.03
        self.weekly_dd_halt: float = 0.07
        self.peak_dd_halt: float = 0.10

        # Signal log
        self.recent_signals: list[dict] = []

        # Alert log
        self.recent_alerts: list[str] = []

        # System health
        self.data_feed_ok: bool = True
        self.api_ok: bool = True
        self.api_latency_ms: float = 0.0
        self.hmm_age_days: int = 0
        self.paper_mode: bool = True

    # Thread-safe setters called from the trading loop
    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def push_signal(self, sig_dict: dict) -> None:
        with self._lock:
            self.recent_signals.append(sig_dict)
            if len(self.recent_signals) > _MAX_SIGNALS:
                self.recent_signals = self.recent_signals[-_MAX_SIGNALS:]

    def push_alert(self, msg: str) -> None:
        with self._lock:
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            self.recent_alerts.append(f"{ts}  {msg}")
            if len(self.recent_alerts) > _MAX_ALERTS:
                self.recent_alerts = self.recent_alerts[-_MAX_ALERTS:]

    def snapshot_copy(self) -> "DashboardState":
        """Return a shallow copy for rendering (avoids holding lock during render)."""
        with self._lock:
            import copy
            return copy.copy(self)


# ─────────────────────────────────────────────────────────────────────────────
# TradingDashboard
# ─────────────────────────────────────────────────────────────────────────────

class TradingDashboard:
    """
    Terminal dashboard rendered via Rich Live.

    Parameters
    ----------
    refresh_seconds : int
        How often to redraw (default 5).
    console : Console | None
        Optional pre-configured console.
    """

    def __init__(
        self,
        refresh_seconds: int = 5,
        console: Optional[Console] = None,
    ) -> None:
        self.refresh_seconds = refresh_seconds
        self.console = console or Console()
        self.state = DashboardState()

        self._live: Optional[Live] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the dashboard refresh loop in a daemon thread (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the refresh loop and restore the terminal."""
        self._running = False
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # ------------------------------------------------------------------
    # State update helpers  (called from trading-loop thread)
    # ------------------------------------------------------------------

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        with self.state._lock:
            self.state.snapshot = snapshot

    def update_regime(
        self,
        label: str,
        probability: float,
        consecutive_bars: int = 0,
        is_confirmed: bool = True,
        flicker_count: int = 0,
        flicker_window: int = 20,
    ) -> None:
        self.state.update(
            regime_label=label,
            regime_probability=probability,
            regime_consecutive_bars=consecutive_bars,
            is_confirmed=is_confirmed,
            flicker_count=flicker_count,
            flicker_window=flicker_window,
        )

    def update_risk(
        self,
        daily_dd: float = 0.0,
        weekly_dd: float = 0.0,
        peak_dd: float = 0.0,
        allocation_pct: float = 0.0,
        leverage: float = 1.0,
    ) -> None:
        self.state.update(
            daily_dd=daily_dd,
            weekly_dd=weekly_dd,
            peak_dd=peak_dd,
            allocation_pct=allocation_pct,
            leverage=leverage,
        )

    def update_system(
        self,
        data_feed_ok: bool = True,
        api_ok: bool = True,
        api_latency_ms: float = 0.0,
        hmm_age_days: int = 0,
        paper_mode: bool = True,
    ) -> None:
        self.state.update(
            data_feed_ok=data_feed_ok,
            api_ok=api_ok,
            api_latency_ms=api_latency_ms,
            hmm_age_days=hmm_age_days,
            paper_mode=paper_mode,
        )

    def push_signal(
        self,
        symbol: str,
        description: str,
        regime: str = "",
    ) -> None:
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M")
        self.state.push_signal({"ts": ts, "symbol": symbol, "desc": description, "regime": regime})

    def add_alert(self, message: str) -> None:
        self.state.push_alert(message)

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        with Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=1,
            screen=False,
        ) as live:
            self._live = live
            while self._running:
                try:
                    live.update(self._build_layout())
                except Exception:
                    pass
                time.sleep(self.refresh_seconds)

    # ------------------------------------------------------------------
    # Layout & render helpers
    # ------------------------------------------------------------------

    def _build_layout(self) -> RenderableType:
        """Assemble the full dashboard as a sequence of panels."""
        s = self.state.snapshot_copy()
        panels: list[RenderableType] = [
            self._render_header(s),
            self._render_regime(s),
            self._render_portfolio(s),
            self._render_positions(s),
            self._render_signals(s),
            self._render_risk_gauges(s),
            self._render_system(s),
        ]
        # Stack vertically
        from rich.console import Group
        return Group(*panels)

    def _render_header(self, s: DashboardState) -> RenderableType:
        now = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
        mode = "[bold yellow]PAPER[/]" if s.paper_mode else "[bold red]⚠  LIVE[/]"
        title = Text.assemble(
            ("  regime-trader  ", "bold cyan"),
            (f"│  {now}  │  ", "white"),
            Text.from_markup(mode),
            ("  ", ""),
        )
        return Panel(title, style="bold cyan", padding=(0, 1))

    def _render_regime(self, s: DashboardState) -> RenderableType:
        colour = _C_REGIME.get(s.regime_label.upper(), "white")
        conf_pct = f"{s.regime_probability * 100:.0f}%"
        confirmed = "✅ confirmed" if s.is_confirmed else "⚠  unconfirmed"
        flicker_ok = s.flicker_count <= 3
        flicker_icon = "✅" if flicker_ok else "⚠ "

        t = Text.assemble(
            (f"  {s.regime_label}", colour),
            (f"  ({conf_pct})", "white"),
            ("  │  ", "dim"),
            (f"Stability: {s.regime_consecutive_bars} bars", "white"),
            ("  │  ", "dim"),
            (f"Flicker: {s.flicker_count}/{s.flicker_window}  {flicker_icon}", "white"),
            ("  │  ", "dim"),
            (confirmed, _C_OK if s.is_confirmed else _C_WARN),
        )
        return Panel(t, title="[bold]REGIME", border_style="green" if s.is_confirmed else "yellow")

    def _render_portfolio(self, s: DashboardState) -> RenderableType:
        snap = s.snapshot
        equity = snap.equity if snap else 0.0
        cash   = snap.cash   if snap else 0.0
        daily_pnl = (snap.realised_pnl_today + snap.unrealised_pnl) if snap else 0.0
        open_eq = s.daily_open_equity or equity or 1.0
        daily_pct = daily_pnl / open_eq * 100 if open_eq else 0.0

        pnl_colour = _C_OK if daily_pnl >= 0 else _C_CRIT
        pnl_sign   = "+" if daily_pnl >= 0 else ""

        row1 = Text.assemble(
            (f"  Equity: ${equity:>12,.2f}", "bold white"),
            ("  │  ", "dim"),
            (f"Daily: {pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pct:.2f}%)", pnl_colour),
        )
        row2 = Text.assemble(
            (f"  Cash: ${cash:>13,.2f}", "white"),
            ("  │  ", "dim"),
            (f"Allocation: {s.allocation_pct:.0f}%", "white"),
            ("  │  ", "dim"),
            (f"Leverage: {s.leverage:.2f}×", "white"),
        )
        from rich.console import Group
        return Panel(Group(row1, row2), title="[bold]PORTFOLIO", border_style="blue")

    def _render_positions(self, s: DashboardState) -> RenderableType:
        snap = s.snapshot
        positions = snap.positions if snap else {}

        tbl = Table(
            "Symbol", "Side", "Price", "P&L", "Unrealised", "Stop", "Holding",
            box=None, padding=(0, 1), show_header=True, header_style="bold dim",
        )

        if not positions:
            tbl.add_row("—", "—", "—", "—", "—", "—", "—")
        else:
            for sym, pos in positions.items():
                pnl_pct = pos.unrealised_pnl_pct * 100
                pnl_col = _C_OK if pnl_pct >= 0 else _C_CRIT
                pnl_sign = "+" if pnl_pct >= 0 else ""

                # Holding time
                holding = getattr(pos, "holding_period_days", 0.0)
                if holding < 1 / 24:
                    h_str = f"{holding * 24 * 60:.0f}m"
                elif holding < 1:
                    h_str = f"{holding * 24:.1f}h"
                else:
                    h_str = f"{holding:.1f}d"

                stop = getattr(pos, "stop_price", 0.0)
                tbl.add_row(
                    f"[bold]{sym}[/]",
                    f"[green]{pos.side.upper()}[/]",
                    f"${pos.current_price:,.2f}",
                    Text(f"{pnl_sign}{pnl_pct:.2f}%", style=pnl_col),
                    f"${pos.unrealised_pnl:+,.2f}",
                    f"${stop:,.2f}" if stop else "—",
                    h_str,
                )

        return Panel(tbl, title="[bold]POSITIONS", border_style="blue")

    def _render_signals(self, s: DashboardState) -> RenderableType:
        lines: list[RenderableType] = []
        with s._lock:
            signals = list(reversed(s.recent_signals))

        if not signals:
            lines.append(Text("  No recent signals.", style="dim"))
        else:
            for sig in signals:
                colour = _C_REGIME.get(sig.get("regime", "").upper(), "white")
                lines.append(Text.assemble(
                    (f"  {sig['ts']}  ", "dim"),
                    (f"{sig['symbol']:6s}", "bold white"),
                    (f"  {sig['desc']}", "white"),
                    (f"  [{sig.get('regime', '')}]", colour),
                ))

        from rich.console import Group
        return Panel(Group(*lines), title="[bold]RECENT SIGNALS", border_style="cyan")

    def _render_risk_gauges(self, s: DashboardState) -> RenderableType:
        def _gauge_line(label: str, current: float, limit: float) -> Text:
            pct = min(current / limit, 1.0) if limit > 0 else 0.0
            bar_len = 20
            filled = int(pct * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            if pct < 0.5:
                bar_colour = "green"
                icon = "✅"
            elif pct < 0.8:
                bar_colour = "yellow"
                icon = "⚠ "
            else:
                bar_colour = "bold red"
                icon = "🔴"
            return Text.assemble(
                (f"  {label:<12}", "white"),
                (f"{current * 100:>5.1f}% / {limit * 100:.1f}%  ", "white"),
                (bar, bar_colour),
                (f"  {icon}", ""),
            )

        from rich.console import Group
        return Panel(
            Group(
                _gauge_line("Daily DD",  s.daily_dd,  s.daily_dd_halt),
                _gauge_line("Weekly DD", s.weekly_dd, s.weekly_dd_halt),
                _gauge_line("Peak DD",   s.peak_dd,   s.peak_dd_halt),
            ),
            title="[bold]RISK STATUS",
            border_style="yellow",
        )

    def _render_system(self, s: DashboardState) -> RenderableType:
        data_icon = "✅" if s.data_feed_ok else "[bold red]❌[/]"
        api_icon  = f"✅ {s.api_latency_ms:.0f}ms" if s.api_ok else "[bold red]❌ LOST[/]"
        hmm_str   = f"{s.hmm_age_days}d ago" if s.hmm_age_days else "fresh"
        mode_str  = "[bold yellow]PAPER[/]" if s.paper_mode else "[bold red]LIVE[/]"

        with s._lock:
            alerts = list(reversed(s.recent_alerts[-3:]))

        alert_lines: list[RenderableType] = [
            Text.assemble(
                (f"  Data: {data_icon}  │  ", "white"),
                Text.from_markup(f"API: {api_icon}  │  "),
                (f"HMM: {hmm_str}  │  ", "white"),
                Text.from_markup(f"Mode: {mode_str}"),
            )
        ]
        for a in alerts:
            alert_lines.append(Text(f"  ⚠  {a}", style="yellow"))

        from rich.console import Group
        return Panel(Group(*alert_lines), title="[bold]SYSTEM", border_style="dim")
