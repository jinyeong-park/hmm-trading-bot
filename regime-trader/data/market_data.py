"""
market_data.py — Historical and real-time market data.

Responsibilities:
  - Pre-fetch historical OHLCV bars for the symbol universe into an in-memory cache.
  - Stream real-time bars and quotes via the Alpaca WebSocket.
  - Expose get_historical_bars(), get_latest_bar(), get_latest_quote(), get_snapshot().
  - Forward-fill gaps (weekends, market holidays, halted sessions).
  - Dispatch new bars to all registered callbacks.
  - Handle gaps gracefully: forward-fill OHLC from prior close, set volume=0.
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta
from typing import Callable, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Timeframe → approximate business-day lookback multiplier
_TF_DAY_MULT = {
    "1min": 390,    # bars per day (6.5 hours × 60)
    "5min": 78,
    "15min": 26,
    "30min": 13,
    "1hour": 7,
    "4hour": 2,
    "1day": 1,
}


class MarketDataFeed:
    """
    Manages historical and streaming OHLCV data for a symbol universe.

    Parameters
    ----------
    client : AlpacaClient
        Authenticated Alpaca API client.
    symbols : list[str]
        Ticker symbols to track.
    timeframe : str
        Bar timeframe, e.g. '1Day', '1Hour', '5Min'.
    lookback_bars : int
        Number of historical bars to pre-fetch and cache per symbol.
    adjustment : str
        Price adjustment: 'raw' | 'split' | 'dividend' | 'all'.
    """

    def __init__(
        self,
        client: AlpacaClient,
        symbols: list[str],
        timeframe: str = "1Day",
        lookback_bars: int = 504,
        adjustment: str = "all",
    ) -> None:
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.adjustment = adjustment

        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._bar_callbacks: list[Callable[[str, pd.Series], None]] = []
        self._quote_callbacks: list[Callable[[dict], None]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Pre-fetch historical bars for all symbols and populate the cache.
        Must be called before starting the streaming feed.
        """
        end = pd.Timestamp.utcnow()
        # Over-fetch by 50% to account for weekends/holidays
        mult = _TF_DAY_MULT.get(self.timeframe.lower().replace(" ", ""), 1)
        calendar_days_needed = int(self.lookback_bars / mult * 1.5) + 30
        start = end - timedelta(days=calendar_days_needed)

        logger.info(
            "Initialising bar cache: %d symbols, %s, lookback=%d bars",
            len(self.symbols), self.timeframe, self.lookback_bars,
        )

        fetched = self.client.get_bars(
            symbols=self.symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            adjustment=self.adjustment,
        )

        with self._lock:
            for sym in self.symbols:
                df = fetched.get(sym, pd.DataFrame(columns=_OHLCV_COLS))
                if len(df) > 0:
                    df = self._fill_gaps(df)
                    # Trim to lookback_bars
                    self._bar_cache[sym] = df.iloc[-self.lookback_bars:].copy()
                    logger.debug("%s: cached %d bars", sym, len(self._bar_cache[sym]))
                else:
                    logger.warning("No historical bars returned for %s", sym)
                    self._bar_cache[sym] = pd.DataFrame(columns=_OHLCV_COLS)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fetch bars directly from Alpaca for an arbitrary date range.

        Parameters
        ----------
        symbol : str
        timeframe : str
        start, end : pd.Timestamp

        Returns
        -------
        pd.DataFrame
            Columns: [open, high, low, close, volume], DatetimeIndex UTC.
        """
        result = self.client.get_bars(
            symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=self.adjustment,
        )
        df = result.get(symbol, pd.DataFrame(columns=_OHLCV_COLS))
        return self._fill_gaps(df) if len(df) > 0 else df

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        Return cached bars for `symbol`.

        Parameters
        ----------
        symbol : str
        n : int | None
            If provided, return the most recent n bars.
        """
        with self._lock:
            df = self._bar_cache.get(symbol, pd.DataFrame(columns=_OHLCV_COLS))
        if n is not None and len(df) >= n:
            return df.iloc[-n:].copy()
        return df.copy()

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """Return the most recent completed bar for `symbol` (dict or None)."""
        return self.client.get_latest_bar(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Return the latest NBBO quote for `symbol` (dict or None)."""
        return self.client.get_latest_quote(symbol)

    def get_snapshot(self, symbols: Optional[list[str]] = None) -> dict[str, dict]:
        """
        Return a full market snapshot (bar + quote + daily stats) for each symbol.

        Parameters
        ----------
        symbols : list[str] | None
            Symbols to snapshot.  Defaults to self.symbols.
        """
        return self.client.get_snapshot(symbols or self.symbols)

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent close price for `symbol`."""
        with self._lock:
            df = self._bar_cache.get(symbol)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        bar = self.client.get_latest_bar(symbol)
        return float(bar["close"]) if bar else 0.0

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def register_bar_callback(self, callback: Callable[[str, pd.Series], None]) -> None:
        """
        Register a function called whenever a new real-time bar arrives.

        Parameters
        ----------
        callback : Callable[[str, pd.Series], None]
            Called with (symbol, bar_as_pd_Series).
        """
        self._bar_callbacks.append(callback)

    def register_quote_callback(self, callback: Callable[[dict], None]) -> None:
        """Register a function called on each real-time NBBO quote update."""
        self._quote_callbacks.append(callback)

    def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: str,
        callback: Callable[[str, pd.Series], None],
    ) -> None:
        """
        Subscribe to real-time bars for `symbols` and register a callback.

        Parameters
        ----------
        symbols : list[str]
        timeframe : str
            Must match the session timeframe — Alpaca streams one timeframe.
        callback : Callable[[str, pd.Series], None]
        """
        self.register_bar_callback(callback)
        self.client.subscribe_bars(symbols, self._on_bar)

    def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to NBBO quotes for `symbols`."""
        self.register_quote_callback(callback)
        self.client.subscribe_quotes(symbols, self._on_quote)

    def start_stream(self) -> None:
        """Start the WebSocket bar/quote stream in a daemon thread."""
        self.client.start_stream()
        logger.info("MarketDataFeed stream started")

    def stop_stream(self) -> None:
        """Stop the WebSocket bar/quote stream."""
        self.client.stop_stream()
        logger.info("MarketDataFeed stream stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_bar(self, raw_bar: dict) -> None:
        """
        WebSocket bar callback: append to cache, notify all registered callbacks.
        """
        symbol = raw_bar.get("symbol", "")
        if not symbol:
            return

        try:
            ts = pd.Timestamp(raw_bar["timestamp"])
            bar = pd.Series(
                {
                    "open": float(raw_bar["open"]),
                    "high": float(raw_bar["high"]),
                    "low": float(raw_bar["low"]),
                    "close": float(raw_bar["close"]),
                    "volume": float(raw_bar["volume"]),
                },
                name=ts,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Malformed bar event for %s: %s", symbol, exc)
            return

        self._append_bar(symbol, bar)

        for cb in self._bar_callbacks:
            try:
                cb(symbol, bar)
            except Exception as exc:  # noqa: BLE001
                logger.error("Bar callback error (%s): %s", symbol, exc)

    def _on_quote(self, quote: dict) -> None:
        """WebSocket quote callback: notify all registered callbacks."""
        for cb in self._quote_callbacks:
            try:
                cb(quote)
            except Exception as exc:  # noqa: BLE001
                logger.error("Quote callback error: %s", exc)

    def _append_bar(self, symbol: str, bar: pd.Series) -> None:
        """
        Append `bar` to the in-memory cache for `symbol`, trim to lookback_bars.
        """
        new_row = pd.DataFrame([bar.values], index=[bar.name], columns=_OHLCV_COLS)

        with self._lock:
            existing = self._bar_cache.get(symbol, pd.DataFrame(columns=_OHLCV_COLS))
            # Drop duplicate timestamps
            if len(existing) > 0 and bar.name in existing.index:
                existing = existing.drop(index=bar.name)
            updated = pd.concat([existing, new_row]).sort_index()
            # Trim to lookback window
            if len(updated) > self.lookback_bars:
                updated = updated.iloc[-self.lookback_bars:]
            self._bar_cache[symbol] = updated

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing bars due to market holidays, early closes, or halts.

        Strategy:
          - Reindex to a complete business-day (or minute/hour) frequency.
          - Forward-fill OHLC from the prior close; set volume = 0 for gap bars
            (no trades occurred).
          - Drop any remaining NaNs (beginning of series before first bar).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
        """
        if len(df) < 2:
            return df

        tf_lower = self.timeframe.lower().replace(" ", "")

        if "day" in tf_lower:
            freq = "B"   # Business day
        elif "hour" in tf_lower or "h" in tf_lower:
            freq = "h"
        elif "min" in tf_lower:
            mins = "".join(c for c in tf_lower if c.isdigit()) or "1"
            freq = f"{mins}min"
        else:
            return df

        try:
            full_index = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=freq,
                tz=df.index.tz,
            )
        except Exception:
            return df

        df_reindexed = df.reindex(full_index)

        # Forward-fill OHLC; new volume = 0 (gap bars had no trades)
        ohlc = df_reindexed[["open", "high", "low", "close"]].ffill()
        volume = df_reindexed["volume"].fillna(0.0)
        filled = pd.concat([ohlc, volume], axis=1)[_OHLCV_COLS]

        # Drop leading NaNs (before first real bar)
        filled = filled.dropna()
        return filled
