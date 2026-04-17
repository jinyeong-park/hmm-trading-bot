"""
alerts.py — Trading event alerts with rate limiting.

Delivery channels:
  - Console  — always active; uses Rich for colour
  - Log file — always active; writes to alerts.log via monitoring.logger
  - Email    — optional; requires SMTP env vars or constructor params
  - Webhook  — optional; posts JSON to Slack / Discord / generic URL

Rate limiting: 1 alert per (event_key, channel) per rate_limit_minutes.
Critical events with force_alert() bypass rate limiting entirely.

Trigger helpers (call from trading loop):
  alert_regime_change(label, prev, probability)
  alert_circuit_breaker(breaker_type, dd, equity)
  alert_large_pnl(symbol, pnl_pct, equity)
  alert_data_feed_down(reason)
  alert_api_lost(reason)
  alert_hmm_retrained(n_states, features_used)
  alert_flicker_exceeded(count, window)

Environment variables (loaded from .env):
  ALERT_EMAIL_HOST, ALERT_EMAIL_PORT, ALERT_EMAIL_USER
  ALERT_EMAIL_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO (comma-separated)
  ALERT_WEBHOOK_URL
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("regime_trader.alerts")

_SEVERITY_COLOURS = {
    "info":     "bold cyan",
    "warning":  "bold yellow",
    "critical": "bold red",
}


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    CONSOLE = "console"
    EMAIL   = "email"
    WEBHOOK = "webhook"
    ALL     = "all"


# ─────────────────────────────────────────────────────────────────────────────
# AlertManager
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Sends and rate-limits trading alerts via console, log, email, and webhook.

    Parameters
    ----------
    smtp_host, smtp_port, smtp_user, smtp_password, from_addr, to_addrs
        SMTP configuration for email delivery.  Can also be loaded from env vars
        (ALERT_EMAIL_HOST etc.) — env vars take priority over constructor params.
    webhook_url : str | None
        Slack / Discord / generic webhook URL.
        Can also be set via ALERT_WEBHOOK_URL env var.
    rate_limit_minutes : int
        Minimum gap between repeated alerts for the same event_key (default 15).
    """

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_addr: str = "",
        to_addrs: Optional[list[str]] = None,
        webhook_url: Optional[str] = None,
        rate_limit_minutes: int = 15,
    ) -> None:
        # Env vars override constructor params
        self.smtp_host     = os.getenv("ALERT_EMAIL_HOST",     smtp_host)
        self.smtp_port     = int(os.getenv("ALERT_EMAIL_PORT", str(smtp_port)))
        self.smtp_user     = os.getenv("ALERT_EMAIL_USER",     smtp_user)
        self.smtp_password = os.getenv("ALERT_EMAIL_PASSWORD", smtp_password)
        self.from_addr     = os.getenv("ALERT_EMAIL_FROM",     from_addr)

        env_to = os.getenv("ALERT_EMAIL_TO", "")
        self.to_addrs = (
            [a.strip() for a in env_to.split(",") if a.strip()]
            if env_to
            else (to_addrs or [])
        )

        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", "") or webhook_url or ""
        self.rate_limit_minutes = rate_limit_minutes

        # event_key → last sent UTC datetime
        self._sent_history: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alert(
        self,
        event_key: str,
        subject: str,
        body: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channel: AlertChannel = AlertChannel.ALL,
        extra: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Send an alert subject to rate limiting.

        Parameters
        ----------
        event_key : str
            Unique key for this alert type (used for rate limiting).
        subject : str
            Short title.
        body : str
            Full message body.
        severity : AlertSeverity
        channel : AlertChannel
        extra : dict | None
            Extra fields appended to webhook payload.

        Returns
        -------
        bool
            True if the alert was dispatched; False if suppressed.
        """
        if self._is_rate_limited(event_key):
            logger.debug("Alert suppressed (rate limit): %s", event_key)
            return False

        self._dispatch(event_key, subject, body, severity, channel, extra)
        self._record_sent(event_key)
        return True

    def force_alert(
        self,
        subject: str,
        body: str,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
        channel: AlertChannel = AlertChannel.ALL,
    ) -> None:
        """
        Send an alert immediately, bypassing rate limiting.
        Use for one-time critical events: startup failure, reconciliation errors.
        """
        self._dispatch("__force__", subject, body, severity, channel, extra=None)

    # ------------------------------------------------------------------
    # Convenience trigger methods
    # ------------------------------------------------------------------

    def alert_regime_change(
        self,
        new_label: str,
        prev_label: str,
        probability: float,
        equity: float = 0.0,
    ) -> bool:
        """Fire when HMM regime changes to a new confirmed label."""
        subject = f"Regime Change: {prev_label} → {new_label}"
        body = (
            f"The HMM regime has changed from {prev_label} to {new_label} "
            f"(confidence {probability:.0%}).\n"
            f"Portfolio equity: ${equity:,.2f}"
        )
        return self.alert(
            event_key=f"regime_change_{new_label}",
            subject=subject,
            body=body,
            severity=AlertSeverity.INFO,
            extra={"new_regime": new_label, "prev_regime": prev_label, "probability": probability},
        )

    def alert_circuit_breaker(
        self,
        breaker_type: str,
        actual_dd: float,
        equity: float,
        positions_closed: Optional[list[str]] = None,
        hmm_regime: str = "UNKNOWN",
    ) -> bool:
        """Fire when a drawdown circuit breaker triggers."""
        subject = f"⚠ Circuit Breaker: {breaker_type.upper()}"
        body = (
            f"Circuit breaker [{breaker_type}] triggered.\n"
            f"Drawdown: {actual_dd:.2%}  |  Equity: ${equity:,.2f}\n"
            f"HMM regime at trigger: {hmm_regime}\n"
        )
        if positions_closed:
            body += f"Positions closed: {', '.join(positions_closed)}"
        severity = (
            AlertSeverity.CRITICAL
            if "halt" in breaker_type.lower()
            else AlertSeverity.WARNING
        )
        return self.alert(
            event_key=f"circuit_breaker_{breaker_type}",
            subject=subject,
            body=body,
            severity=severity,
            extra={"breaker_type": breaker_type, "actual_dd": actual_dd, "equity": equity},
        )

    def alert_large_pnl(
        self,
        symbol: str,
        pnl_pct: float,
        equity: float,
        threshold_pct: float = 2.0,
    ) -> bool:
        """Fire when a single-position P&L exceeds ±threshold_pct."""
        direction = "gain" if pnl_pct >= 0 else "loss"
        sign = "+" if pnl_pct >= 0 else ""
        subject = f"Large P&L {direction}: {symbol} {sign}{pnl_pct:.1f}%"
        body = (
            f"{symbol} has a {direction} of {sign}{pnl_pct:.2f}%.\n"
            f"Current portfolio equity: ${equity:,.2f}"
        )
        severity = (
            AlertSeverity.WARNING
            if abs(pnl_pct) < threshold_pct * 2
            else AlertSeverity.CRITICAL
        )
        return self.alert(
            event_key=f"large_pnl_{symbol}",
            subject=subject,
            body=body,
            severity=severity,
            extra={"symbol": symbol, "pnl_pct": pnl_pct},
        )

    def alert_data_feed_down(self, reason: str = "") -> bool:
        """Fire when the WebSocket data feed drops."""
        subject = "⚠ Data Feed Down"
        body = f"WebSocket data feed is unavailable.\nSignals paused; stops remain active.\n{reason}"
        return self.alert(
            event_key="data_feed_down",
            subject=subject,
            body=body,
            severity=AlertSeverity.CRITICAL,
        )

    def alert_api_lost(self, reason: str = "") -> bool:
        """Fire when the Alpaca REST API becomes unreachable."""
        subject = "⚠ Alpaca API Unreachable"
        body = f"Lost connection to Alpaca API.\nNo new orders will be submitted.\n{reason}"
        return self.alert(
            event_key="api_lost",
            subject=subject,
            body=body,
            severity=AlertSeverity.CRITICAL,
        )

    def alert_hmm_retrained(self, n_states: int, features_used: int) -> bool:
        """Fire after a successful weekly HMM retrain."""
        subject = f"HMM Retrained: {n_states} states"
        body = (
            f"Weekly HMM retrain complete.\n"
            f"n_states={n_states}  training_bars={features_used}"
        )
        return self.alert(
            event_key="hmm_retrained",
            subject=subject,
            body=body,
            severity=AlertSeverity.INFO,
        )

    def alert_flicker_exceeded(self, count: int, window: int, threshold: int) -> bool:
        """Fire when the regime flicker rate exceeds the configured threshold."""
        subject = f"Regime Flicker Exceeded: {count}/{window}"
        body = (
            f"Regime flicker rate {count}/{window} exceeded threshold {threshold}.\n"
            "Uncertainty mode active — position sizes halved."
        )
        return self.alert(
            event_key="flicker_exceeded",
            subject=subject,
            body=body,
            severity=AlertSeverity.WARNING,
            extra={"count": count, "window": window, "threshold": threshold},
        )

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str, severity: AlertSeverity) -> None:
        """Deliver an email via SMTP TLS."""
        if not self.smtp_host or not self.to_addrs:
            return
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[regime-trader] {subject}"
            msg["From"]    = self.from_addr
            msg["To"]      = ", ".join(self.to_addrs)
            severity_tag   = severity.value.upper()

            text_body = f"[{severity_tag}] {subject}\n\n{body}"
            html_body = (
                f"<html><body>"
                f"<h3>[{severity_tag}] {subject}</h3>"
                f"<pre>{body}</pre>"
                f"<hr><small>regime-trader alert  ·  {datetime.now(tz=timezone.utc).isoformat()}</small>"
                f"</body></html>"
            )
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.debug("Email sent to %s: %s", self.to_addrs, subject)
        except Exception as exc:
            logger.warning("Email delivery failed: %s", exc)

    def _send_webhook(
        self,
        subject: str,
        body: str,
        severity: AlertSeverity,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """POST a JSON payload to self.webhook_url."""
        if not self.webhook_url:
            return
        try:
            # Slack / Discord colour map
            colour_map = {
                AlertSeverity.INFO:     "#36a64f",
                AlertSeverity.WARNING:  "#ffcc00",
                AlertSeverity.CRITICAL: "#ff0000",
            }

            payload: dict[str, Any] = {
                "text": f"*[{severity.value.upper()}]* {subject}",
                "attachments": [
                    {
                        "color": colour_map.get(severity, "#cccccc"),
                        "text": body,
                        "footer": f"regime-trader  ·  {datetime.now(tz=timezone.utc).isoformat()}",
                    }
                ],
            }
            if extra:
                payload["extra"] = extra

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status >= 400:
                    logger.warning("Webhook returned HTTP %d", resp.status)
                else:
                    logger.debug("Webhook delivered: %s", subject)
        except Exception as exc:
            logger.warning("Webhook delivery failed: %s", exc)

    def _send_console(self, subject: str, body: str, severity: AlertSeverity) -> None:
        """Print a colour-coded alert to stdout."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            c = Console()
            colour = {
                AlertSeverity.INFO:     "cyan",
                AlertSeverity.WARNING:  "yellow",
                AlertSeverity.CRITICAL: "bold red",
            }.get(severity, "white")
            c.print(Panel(
                f"{body}",
                title=f"[{colour}][{severity.value.upper()}] {subject}[/{colour}]",
                border_style=colour,
            ))
        except ImportError:
            print(f"\n[{severity.value.upper()}] {subject}\n{body}\n")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, event_key: str) -> bool:
        last = self._sent_history.get(event_key)
        if last is None:
            return False
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=self.rate_limit_minutes)
        return last > cutoff

    def _record_sent(self, event_key: str) -> None:
        self._sent_history[event_key] = datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        event_key: str,
        subject: str,
        body: str,
        severity: AlertSeverity,
        channel: AlertChannel,
        extra: Optional[dict[str, Any]],
    ) -> None:
        """Route alert to the configured channel(s) and always log."""
        from monitoring.logger import alert_event
        alert_event(event_key, subject, body, severity.value)

        send_all = channel == AlertChannel.ALL
        if send_all or channel == AlertChannel.CONSOLE:
            self._send_console(subject, body, severity)
        if send_all or channel == AlertChannel.EMAIL:
            self._send_email(subject, body, severity)
        if send_all or channel == AlertChannel.WEBHOOK:
            self._send_webhook(subject, body, severity, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton  (convenience)
# ─────────────────────────────────────────────────────────────────────────────

_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Return the module-level AlertManager singleton.
    Reads configuration from environment variables on first call.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


def configure_alerts(
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
    from_addr: str = "",
    to_addrs: Optional[list[str]] = None,
    webhook_url: Optional[str] = None,
    rate_limit_minutes: int = 15,
) -> AlertManager:
    """
    Configure and return the module-level AlertManager singleton.
    Overwrites any previous singleton.
    """
    global _default_manager
    _default_manager = AlertManager(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_addr=from_addr,
        to_addrs=to_addrs,
        webhook_url=webhook_url,
        rate_limit_minutes=rate_limit_minutes,
    )
    return _default_manager
