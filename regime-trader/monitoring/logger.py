"""
logger.py — Structured logging configuration for regime-trader.

Responsibilities:
  - Configure root logger with rotating file handler and rich console handler.
  - Provide a factory function get_logger() for per-module loggers.
  - Support structured JSON log lines for machine parsing.
  - Expose a trade_event() helper that always logs to a dedicated trades log.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Optional


def configure_logging(
    log_dir: str = "logs/",
    log_level: str = "INFO",
    rotation_mb: int = 10,
    backup_count: int = 5,
    json_format: bool = False,
) -> None:
    """
    Set up root logger with a rotating file handler and a Rich console handler.

    Should be called once at application startup (main.py).

    Parameters
    ----------
    log_dir : str
        Directory where log files are written.
    log_level : str
        Logging level name: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'.
    rotation_mb : int
        File size limit before rotation (megabytes).
    backup_count : int
        Number of rotated files to retain.
    json_format : bool
        If True, emit JSON-formatted log records instead of human-readable text.
    """
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger inheriting from the root configuration.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
    """
    raise NotImplementedError


def trade_event(
    action: str,
    symbol: str,
    qty: float,
    price: float,
    regime: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Log a structured trade event to both the main log and a dedicated trades log.

    Parameters
    ----------
    action : str
        'buy' | 'sell' | 'cancel' | 'rebalance'
    symbol : str
    qty : float
    price : float
    regime : str
        Active regime label at time of trade.
    extra : dict | None
        Additional key-value pairs to include in the structured record.
    """
    raise NotImplementedError


class _JsonFormatter(logging.Formatter):
    """
    Custom log formatter that serialises LogRecord to a JSON string.
    Includes: timestamp, level, name, message, and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        raise NotImplementedError
