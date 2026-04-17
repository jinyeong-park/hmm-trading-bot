"""
logger.py — Structured JSON logging for regime-trader.

Four rotating log files (10 MB each, 30-day backup window):
  main.log    — all application events
  trades.log  — trade fills, submissions, cancellations
  alerts.log  — alert triggers and dispatches
  regime.log  — regime-change events and HMM predictions

Every JSON record includes: timestamp, level, name, message,
plus context fields: regime, probability, equity, positions, daily_pnl.

Usage
-----
    from monitoring.logger import configure_logging, get_logger, trade_event

    configure_logging(log_dir="logs/", log_level="INFO", json_format=True)
    log = get_logger(__name__)
    log.info("Signal approved", extra={"regime": "BULL", "equity": 105230})
    trade_event("buy", "SPY", 10, 521.30, "BULL")
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Module-level shared state
# ─────────────────────────────────────────────────────────────────────────────

_configured = False
_configure_lock = threading.Lock()

# Per-file loggers
_trade_logger:  Optional[logging.Logger] = None
_alert_logger:  Optional[logging.Logger] = None
_regime_logger: Optional[logging.Logger] = None

# Context injected into every record
_global_context: dict[str, Any] = {
    "regime": "UNKNOWN",
    "probability": 0.0,
    "equity": 0.0,
    "positions": [],
    "daily_pnl": 0.0,
}
_context_lock = threading.Lock()

_BACKUP_COUNT = 30        # ~30 days at 1 rotation per day
_ROTATION_BYTES = 10 * 1024 * 1024   # 10 MB


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(
    log_dir: str = "logs/",
    log_level: str = "INFO",
    rotation_mb: int = 10,
    backup_count: int = _BACKUP_COUNT,
    json_format: bool = True,
) -> None:
    """
    Set up root logger with rotating file + Rich (or plain) console handlers.

    Idempotent — safe to call multiple times; only configures once.

    Parameters
    ----------
    log_dir : str
        Directory where log files are written.
    log_level : str
        'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
    rotation_mb : int
        File size limit before rotation (megabytes).
    backup_count : int
        Number of rotated files to keep.
    json_format : bool
        Emit JSON log records when True; human-readable text when False.
    """
    global _configured, _trade_logger, _alert_logger, _regime_logger

    with _configure_lock:
        if _configured:
            return

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        rotation_bytes = rotation_mb * 1024 * 1024

        formatter: logging.Formatter = (
            _JsonFormatter() if json_format else _PlainFormatter()
        )

        # ── Root / main logger ────────────────────────────────────────────
        root = logging.getLogger()
        root.setLevel(numeric_level)
        root.handlers.clear()

        # Console handler
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                level=numeric_level,
                show_time=True,
                show_path=False,
                markup=True,
            )
        except ImportError:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root.addHandler(console_handler)

        # main.log
        main_fh = _make_rotating_handler(
            log_path / "main.log", rotation_bytes, backup_count, formatter, numeric_level
        )
        root.addHandler(main_fh)

        # ── trades.log ────────────────────────────────────────────────────
        _trade_logger = logging.getLogger("regime_trader.trades")
        _trade_logger.propagate = False
        _trade_logger.setLevel(logging.DEBUG)
        _trade_logger.handlers.clear()
        _trade_logger.addHandler(
            _make_rotating_handler(
                log_path / "trades.log", rotation_bytes, backup_count, formatter, logging.DEBUG
            )
        )

        # ── alerts.log ────────────────────────────────────────────────────
        _alert_logger = logging.getLogger("regime_trader.alerts")
        _alert_logger.propagate = False
        _alert_logger.setLevel(logging.DEBUG)
        _alert_logger.handlers.clear()
        _alert_logger.addHandler(
            _make_rotating_handler(
                log_path / "alerts.log", rotation_bytes, backup_count, formatter, logging.DEBUG
            )
        )

        # ── regime.log ────────────────────────────────────────────────────
        _regime_logger = logging.getLogger("regime_trader.regime")
        _regime_logger.propagate = False
        _regime_logger.setLevel(logging.DEBUG)
        _regime_logger.handlers.clear()
        _regime_logger.addHandler(
            _make_rotating_handler(
                log_path / "regime.log", rotation_bytes, backup_count, formatter, logging.DEBUG
            )
        )

        _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger inheriting the root configuration.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.
    """
    return logging.getLogger(name)


def set_context(
    regime: Optional[str] = None,
    probability: Optional[float] = None,
    equity: Optional[float] = None,
    positions: Optional[list] = None,
    daily_pnl: Optional[float] = None,
) -> None:
    """
    Update the global context injected into every JSON log record.

    Call once per bar from the trading loop.
    """
    with _context_lock:
        if regime is not None:
            _global_context["regime"] = regime
        if probability is not None:
            _global_context["probability"] = probability
        if equity is not None:
            _global_context["equity"] = equity
        if positions is not None:
            _global_context["positions"] = positions
        if daily_pnl is not None:
            _global_context["daily_pnl"] = daily_pnl


def trade_event(
    action: str,
    symbol: str,
    qty: float,
    price: float,
    regime: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Emit a structured trade event to trades.log (and main.log via propagation-off logger).

    Parameters
    ----------
    action : str
        'buy' | 'sell' | 'cancel' | 'rebalance'
    symbol : str
    qty : float
    price : float
    regime : str
    extra : dict | None
    """
    if _trade_logger is None:
        # Fallback if configure_logging() was never called
        logging.getLogger("regime_trader.trades").info(
            "TRADE %s %s qty=%.4f @ %.4f  regime=%s", action, symbol, qty, price, regime
        )
        return

    record: dict[str, Any] = {
        "action": action,
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "regime": regime,
        "notional": round(qty * price, 2),
    }
    if extra:
        record.update(extra)

    _trade_logger.info(
        "TRADE %s %s qty=%.4f @ %.4f",
        action.upper(), symbol, qty, price,
        extra={"trade": record},
    )


def regime_event(
    label: str,
    probability: float,
    previous_label: str,
    consecutive_bars: int,
    is_confirmed: bool,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Emit a structured regime-change event to regime.log.

    Parameters
    ----------
    label : str
    probability : float
    previous_label : str
    consecutive_bars : int
    is_confirmed : bool
    extra : dict | None
    """
    if _regime_logger is None:
        logging.getLogger("regime_trader.regime").info(
            "REGIME %s → %s (conf=%.0f%% bars=%d confirmed=%s)",
            previous_label, label, probability * 100, consecutive_bars, is_confirmed,
        )
        return

    record: dict[str, Any] = {
        "label": label,
        "probability": probability,
        "previous_label": previous_label,
        "consecutive_bars": consecutive_bars,
        "is_confirmed": is_confirmed,
    }
    if extra:
        record.update(extra)

    _regime_logger.info(
        "REGIME %s → %s (conf=%.0f%%  bars=%d  confirmed=%s)",
        previous_label, label, probability * 100, consecutive_bars, is_confirmed,
        extra={"regime_event": record},
    )


def alert_event(
    event_key: str,
    subject: str,
    body: str,
    severity: str,
) -> None:
    """Write a structured alert record to alerts.log."""
    if _alert_logger is None:
        logging.getLogger("regime_trader.alerts").warning(
            "ALERT [%s] %s: %s", severity.upper(), subject, body
        )
        return

    _alert_logger.warning(
        "ALERT [%s] %s",
        severity.upper(), subject,
        extra={
            "alert": {
                "event_key": event_key,
                "subject": subject,
                "body": body,
                "severity": severity,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    Serialize LogRecord to a single JSON line.

    Every record includes the global context fields (regime, probability,
    equity, positions, daily_pnl) plus any `extra` dict passed to the logger.
    """

    def format(self, record: logging.LogRecord) -> str:
        with _context_lock:
            ctx = dict(_global_context)

        doc: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            # Global context
            "regime":    ctx["regime"],
            "probability": ctx["probability"],
            "equity":    ctx["equity"],
            "positions": ctx["positions"],
            "daily_pnl": ctx["daily_pnl"],
        }

        # Merge any extra dict-valued fields attached by the caller
        for key in ("trade", "regime_event", "alert"):
            val = getattr(record, key, None)
            if val is not None:
                doc[key] = val

        if record.exc_info:
            doc["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(doc, default=str)


class _PlainFormatter(logging.Formatter):
    """Human-readable formatter for development use."""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rotating_handler(
    path: Path,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    level: int,
) -> logging.handlers.RotatingFileHandler:
    handler = logging.handlers.RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler
