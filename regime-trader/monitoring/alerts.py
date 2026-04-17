"""
alerts.py — Email and webhook alerts for critical trading events.

Responsibilities:
  - Send email alerts via SMTP for risk halts, large drawdowns, order failures.
  - Post to Slack / Discord / generic webhooks for real-time notifications.
  - Rate-limit repeated alerts to avoid notification storms.
  - Provide a unified alert() interface so callers don't need to know the transport.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    BOTH = "both"


class AlertManager:
    """
    Sends and rate-limits trading alerts via email and/or webhooks.

    Parameters
    ----------
    smtp_host : str
    smtp_port : int
    smtp_user : str
    smtp_password : str
    from_addr : str
    to_addrs : list[str]
    webhook_url : str | None
        Slack / Discord / PagerDuty webhook URL. If None, webhook channel is skipped.
    rate_limit_minutes : int
        Minimum gap between repeated alerts for the same event key.
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
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.webhook_url = webhook_url
        self.rate_limit_minutes = rate_limit_minutes

        self._sent_history: dict[str, datetime] = {}   # event_key → last sent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alert(
        self,
        event_key: str,
        subject: str,
        body: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channel: AlertChannel = AlertChannel.BOTH,
        extra: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Send an alert through the configured channel(s), subject to rate limiting.

        Parameters
        ----------
        event_key : str
            Unique key identifying this type of alert (used for rate limiting).
        subject : str
            Email subject / webhook title.
        body : str
            Alert message body.
        severity : AlertSeverity
        channel : AlertChannel
        extra : dict | None
            Additional fields to include in the webhook payload.

        Returns
        -------
        bool
            True if the alert was sent, False if suppressed by rate limiting.
        """
        raise NotImplementedError

    def force_alert(
        self,
        subject: str,
        body: str,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
        channel: AlertChannel = AlertChannel.BOTH,
    ) -> None:
        """
        Send an alert immediately, bypassing rate limiting.
        Use for one-time critical events (startup failures, position reconciliation errors).

        Parameters
        ----------
        subject : str
        body : str
        severity : AlertSeverity
        channel : AlertChannel
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str) -> None:
        """
        Deliver an email via SMTP with TLS.

        Parameters
        ----------
        subject : str
        body : str
        """
        raise NotImplementedError

    def _send_webhook(
        self,
        subject: str,
        body: str,
        severity: AlertSeverity,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        POST a JSON payload to self.webhook_url.

        Parameters
        ----------
        subject : str
        body : str
        severity : AlertSeverity
        extra : dict | None
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, event_key: str) -> bool:
        """
        Return True if an alert for `event_key` was sent within rate_limit_minutes.

        Parameters
        ----------
        event_key : str

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def _record_sent(self, event_key: str) -> None:
        """Record that an alert was just sent for `event_key`."""
        raise NotImplementedError
