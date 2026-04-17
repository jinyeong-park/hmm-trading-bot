"""
alpaca_client.py — Alpaca API wrapper.

Responsibilities:
  - Initialize authenticated Alpaca REST and streaming clients from env / config.
  - Expose account info, clock, and calendar queries.
  - Provide a thin abstraction over both alpaca-py (v2) and alpaca-trade-api (v1)
    so the rest of the codebase stays broker-agnostic.
  - Handle rate limiting and transient HTTP errors with exponential backoff.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


class AlpacaClient:
    """
    Thin wrapper around the Alpaca brokerage API.

    Reads credentials from environment variables:
      ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER (true/false)

    Parameters
    ----------
    paper : bool | None
        Override for paper/live trading. If None, reads from ALPACA_PAPER env var.
    """

    def __init__(self, paper: Optional[bool] = None) -> None:
        self.paper = paper
        self._rest_client: Any = None      # alpaca_trade_api.REST or alpaca.TradingClient
        self._data_client: Any = None      # alpaca.StockHistoricalDataClient
        self._stream_client: Any = None    # alpaca.StockDataStream

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialise REST and data clients using credentials from the environment.

        Raises
        ------
        EnvironmentError
            If ALPACA_API_KEY or ALPACA_SECRET_KEY are not set.
        ConnectionError
            If the initial account ping fails.
        """
        raise NotImplementedError

    def disconnect(self) -> None:
        """Close any open streaming connections gracefully."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Account & market status
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """
        Return account details (equity, cash, buying power, etc.).

        Returns
        -------
        dict[str, Any]
        """
        raise NotImplementedError

    def get_equity(self) -> float:
        """Return current portfolio equity as float USD."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        raise NotImplementedError

    def get_next_market_open(self) -> pd.Timestamp:
        """Return the UTC timestamp of the next market open."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjustment: str = "all",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for one or more symbols.

        Parameters
        ----------
        symbols : list[str]
        timeframe : str
            Alpaca timeframe string, e.g. '1Day', '1Hour', '5Min'.
        start : pd.Timestamp
        end : pd.Timestamp
        adjustment : str
            'raw' | 'split' | 'dividend' | 'all'

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping symbol → DataFrame with columns [open, high, low, close, volume].
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def subscribe_bars(
        self,
        symbols: list[str],
        callback,  # Callable[[dict], None]
    ) -> None:
        """
        Subscribe to real-time bar updates for `symbols`.

        Parameters
        ----------
        symbols : list[str]
        callback : Callable[[dict], None]
            Called with each incoming bar dict.
        """
        raise NotImplementedError

    def start_stream(self) -> None:
        """Start the WebSocket data stream in a background thread."""
        raise NotImplementedError

    def stop_stream(self) -> None:
        """Stop the WebSocket data stream."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_credentials(self) -> tuple[str, str]:
        """
        Read API key and secret from environment variables.

        Returns
        -------
        tuple[str, str]
            (api_key, secret_key)

        Raises
        ------
        EnvironmentError
        """
        raise NotImplementedError

    def _retry_request(self, func, *args, max_retries: int = 3, **kwargs) -> Any:
        """
        Call `func(*args, **kwargs)` with exponential backoff on transient errors.

        Parameters
        ----------
        func : Callable
        max_retries : int

        Returns
        -------
        Any
            Return value of func.
        """
        raise NotImplementedError
