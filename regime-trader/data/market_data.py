"""
market_data.py — Real-time and historical market data fetching.

Responsibilities:
  - Fetch historical OHLCV bars for the configured symbol universe.
  - Subscribe to real-time bar updates via the Alpaca WebSocket stream.
  - Maintain an in-memory bar cache per symbol for fast feature computation.
  - Handle data gaps and corporate actions (splits, dividends).
  - Provide a unified DataFrame interface regardless of data source.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient


class MarketDataFeed:
    """
    Manages historical and streaming OHLCV data for a symbol universe.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client.
    symbols : list[str]
        List of ticker symbols to track.
    timeframe : str
        Bar timeframe, e.g. '1Day', '1Hour', '5Min'.
    lookback_bars : int
        Number of historical bars to pre-fetch and cache per symbol.
    """

    def __init__(
        self,
        client: AlpacaClient,
        symbols: list[str],
        timeframe: str = "1Day",
        lookback_bars: int = 504,
    ) -> None:
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars

        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._callbacks: list[Callable[[str, pd.Series], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Pre-fetch historical bars for all symbols and populate the cache.
        Must be called before starting the streaming feed.
        """
        raise NotImplementedError

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        Return cached bars for `symbol`.

        Parameters
        ----------
        symbol : str
        n : int | None
            If provided, return only the most recent n bars.

        Returns
        -------
        pd.DataFrame
            Columns: [open, high, low, close, volume], DatetimeIndex.
        """
        raise NotImplementedError

    def get_latest_price(self, symbol: str) -> float:
        """
        Return the most recent close price for `symbol`.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def register_bar_callback(self, callback: Callable[[str, pd.Series], None]) -> None:
        """
        Register a function to be called whenever a new real-time bar arrives.

        Parameters
        ----------
        callback : Callable[[str, pd.Series], None]
            Called with (symbol, bar_series) for each incoming bar.
        """
        raise NotImplementedError

    def start_stream(self) -> None:
        """Start the WebSocket bar stream in a daemon thread."""
        raise NotImplementedError

    def stop_stream(self) -> None:
        """Stop the WebSocket bar stream."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_historical(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fetch and clean historical OHLCV bars from Alpaca.

        Parameters
        ----------
        symbol : str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError

    def _on_bar(self, raw_bar: dict) -> None:
        """
        WebSocket callback: append the incoming bar to the cache and notify
        registered callbacks.

        Parameters
        ----------
        raw_bar : dict
            Raw bar event from Alpaca stream.
        """
        raise NotImplementedError

    def _append_bar(self, symbol: str, bar: pd.Series) -> None:
        """
        Append `bar` to the in-memory cache for `symbol` and trim to lookback_bars.

        Parameters
        ----------
        symbol : str
        bar : pd.Series
            Index includes: open, high, low, close, volume.
        """
        raise NotImplementedError

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing bars in `df` (market holidays, early closes).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError
