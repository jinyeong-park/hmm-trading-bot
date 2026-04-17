"""
alpaca_client.py — Alpaca API wrapper (alpaca-py SDK).

Responsibilities:
  - Load credentials from .env (ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER).
  - Gate live-trading behind a manual confirmation prompt.
  - Expose account, positions, clock, and order history queries.
  - Wrap every outbound call in exponential-backoff retry logic.
  - Manage the streaming connection lifecycle in a daemon thread.

All alpaca-py imports are deferred so the module imports cleanly even when
alpaca-py is not installed (useful in backtest-only environments).

Environment variables (loaded from .env automatically):
  ALPACA_API_KEY       — required
  ALPACA_SECRET_KEY    — required
  ALPACA_PAPER         — "true" | "false"  (default: "true")
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_PAPER = "https://paper-api.alpaca.markets"
_BASE_LIVE = "https://api.alpaca.markets"
_DATA_URL = "https://data.alpaca.markets"

_LIVE_CONFIRM_PHRASE = "YES I UNDERSTAND THE RISKS"

# Retry config
_MAX_RETRIES = 3
_BACKOFF_BASE = 2       # seconds; sleep = base ** attempt, capped at 30s
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class AlpacaClient:
    """
    Thin wrapper around the alpaca-py brokerage and data APIs.

    Parameters
    ----------
    paper : bool | None
        Override for paper / live mode.  If None, reads ALPACA_PAPER env var.
        Defaults to paper=True when the env var is absent.
    """

    def __init__(self, paper: Optional[bool] = None) -> None:
        self.paper = paper
        self._trading_client: Any = None    # alpaca.trading.client.TradingClient
        self._data_client: Any = None       # alpaca.data.historical.StockHistoricalDataClient
        self._trading_stream: Any = None    # alpaca.trading.stream.TradingStream
        self._data_stream: Any = None       # alpaca.data.live.StockDataStream
        self._stream_thread: Optional[threading.Thread] = None
        self._is_connected: bool = False
        self._api_key: str = ""
        self._secret_key: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialise REST and data clients using credentials from the environment.

        Raises
        ------
        EnvironmentError
            If ALPACA_API_KEY or ALPACA_SECRET_KEY are missing.
        RuntimeError
            If live mode is not explicitly confirmed.
        ConnectionError
            If the initial health-check ping fails.
        """
        self._api_key, self._secret_key = self._load_credentials()

        # Resolve paper/live mode
        if self.paper is None:
            raw = os.getenv("ALPACA_PAPER", "true").strip().lower()
            self.paper = raw not in ("false", "0", "no")

        if not self.paper:
            self._confirm_live_trading()

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from exc

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self.paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        # Health check — raises ConnectionError on failure
        try:
            self._retry_request(self._trading_client.get_account)
        except Exception as exc:
            raise ConnectionError(
                f"Alpaca health check failed: {exc}"
            ) from exc

        self._is_connected = True
        mode = "paper" if self.paper else "LIVE"
        logger.info("AlpacaClient connected (%s mode)", mode)

    def disconnect(self) -> None:
        """Close any open streaming connections gracefully."""
        self._stop_trading_stream()
        self._stop_data_stream()
        self._is_connected = False
        logger.info("AlpacaClient disconnected")

    def _check_connected(self) -> None:
        if not self._is_connected:
            raise RuntimeError("AlpacaClient not connected — call connect() first")

    # ------------------------------------------------------------------
    # Account & market status
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Return account details: equity, cash, buying_power, margin, etc."""
        self._check_connected()
        acc = self._retry_request(self._trading_client.get_account)
        return {
            "equity": float(acc.equity or 0),
            "cash": float(acc.cash or 0),
            "buying_power": float(acc.buying_power or 0),
            "portfolio_value": float(acc.portfolio_value or 0),
            "long_market_value": float(acc.long_market_value or 0),
            "short_market_value": float(acc.short_market_value or 0),
            "maintenance_margin": float(acc.maintenance_margin or 0),
            "regt_buying_power": float(acc.regt_buying_power or 0),
            "day_trade_count": int(acc.daytrade_count or 0),
            "pattern_day_trader": bool(acc.pattern_day_trader),
            "trading_blocked": bool(acc.trading_blocked),
            "status": str(acc.status),
        }

    def get_equity(self) -> float:
        """Return current portfolio equity in USD."""
        return self.get_account()["equity"]

    def get_available_margin(self) -> float:
        """Return Regulation T buying power (margin-adjusted available funds)."""
        return self.get_account()["regt_buying_power"]

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions as a list of dicts."""
        self._check_connected()
        raw = self._retry_request(self._trading_client.get_all_positions)
        positions = []
        for p in raw:
            positions.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price or p.avg_entry_price),
                "market_value": float(p.market_value or 0),
                "unrealised_pl": float(p.unrealized_pl or 0),
                "unrealised_plpc": float(p.unrealized_plpc or 0),
                "side": str(p.side),
                "asset_class": str(p.asset_class),
            })
        return positions

    def get_order_history(
        self,
        status: str = "closed",
        limit: int = 100,
        after: Optional[pd.Timestamp] = None,
    ) -> list[dict[str, Any]]:
        """
        Return past orders filtered by status.

        Parameters
        ----------
        status : str
            'open' | 'closed' | 'all'
        limit : int
        after : pd.Timestamp | None
            Only return orders submitted after this time.
        """
        self._check_connected()
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(
            status=status_map.get(status, QueryOrderStatus.CLOSED),
            limit=limit,
            after=after.isoformat() if after is not None else None,
        )
        raw = self._retry_request(self._trading_client.get_orders, req)
        orders = []
        for o in raw:
            orders.append({
                "id": str(o.id),
                "client_order_id": str(o.client_order_id),
                "symbol": o.symbol,
                "side": str(o.side),
                "type": str(o.order_type),
                "qty": float(o.qty or 0),
                "filled_qty": float(o.filled_qty or 0),
                "filled_avg_price": float(o.filled_avg_price or 0),
                "status": str(o.status),
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "limit_price": float(o.limit_price or 0),
                "stop_price": float(o.stop_price or 0),
            })
        return orders

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        self._check_connected()
        clock = self._retry_request(self._trading_client.get_clock)
        return bool(clock.is_open)

    def get_clock(self) -> dict[str, Any]:
        """Return market clock: timestamp, is_open, next_open, next_close."""
        self._check_connected()
        clock = self._retry_request(self._trading_client.get_clock)
        return {
            "timestamp": str(clock.timestamp),
            "is_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    def get_next_market_open(self) -> pd.Timestamp:
        """Return the UTC timestamp of the next market open."""
        clock_data = self.get_clock()
        return pd.Timestamp(clock_data["next_open"])

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
            '1Day' | '1Hour' | '5Min' | '1Min'
        start, end : pd.Timestamp
        adjustment : str
            'raw' | 'split' | 'dividend' | 'all'

        Returns
        -------
        dict[str, pd.DataFrame]
            symbol → DataFrame[open, high, low, close, volume], DatetimeIndex UTC
        """
        self._check_connected()
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        tf = _parse_timeframe(timeframe)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start.isoformat(),
            end=end.isoformat(),
            adjustment=adjustment,
        )
        raw = self._retry_request(self._data_client.get_stock_bars, req)

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = raw[sym].df
                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = "timestamp"
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                result[sym] = df.sort_index()
            except (KeyError, AttributeError):
                logger.warning("No bars returned for %s in [%s, %s]", sym, start, end)
                result[sym] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
        return result

    def get_latest_bar(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the most recent completed bar for `symbol`."""
        self._check_connected()
        try:
            from alpaca.data.requests import StockLatestBarRequest
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        req = StockLatestBarRequest(symbol_or_symbols=symbol)
        raw = self._retry_request(self._data_client.get_stock_latest_bar, req)
        bar = raw.get(symbol)
        if bar is None:
            return None
        return {
            "symbol": symbol,
            "timestamp": str(bar.timestamp),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }

    def get_latest_quote(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the latest NBBO quote (bid/ask) for `symbol`."""
        self._check_connected()
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        raw = self._retry_request(self._data_client.get_stock_latest_quote, req)
        q = raw.get(symbol)
        if q is None:
            return None
        bid = float(q.bid_price or 0)
        ask = float(q.ask_price or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_size": float(q.bid_size or 0),
            "ask_size": float(q.ask_size or 0),
            "spread_pct": spread_pct,
            "timestamp": str(q.timestamp),
        }

    def get_snapshot(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """
        Return a full snapshot (latest bar + quote + daily stats) for each symbol.
        """
        self._check_connected()
        try:
            from alpaca.data.requests import StockSnapshotRequest
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        req = StockSnapshotRequest(symbol_or_symbols=symbols)
        raw = self._retry_request(self._data_client.get_stock_snapshot, req)

        result: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            snap = raw.get(sym)
            if snap is None:
                continue
            result[sym] = {
                "symbol": sym,
                "latest_trade_price": float(snap.latest_trade.price if snap.latest_trade else 0),
                "bid": float(snap.latest_quote.bid_price if snap.latest_quote else 0),
                "ask": float(snap.latest_quote.ask_price if snap.latest_quote else 0),
                "prev_close": float(snap.prev_daily_bar.close if snap.prev_daily_bar else 0),
                "daily_bar_open": float(snap.daily_bar.open if snap.daily_bar else 0),
                "daily_bar_close": float(snap.daily_bar.close if snap.daily_bar else 0),
            }
        return result

    # ------------------------------------------------------------------
    # Streaming — data (bars, quotes)
    # ------------------------------------------------------------------

    def subscribe_bars(
        self,
        symbols: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """
        Subscribe to real-time bar updates.

        Parameters
        ----------
        symbols : list[str]
        callback : Callable[[dict], None]
            Called with a normalised bar dict on each incoming bar.
        """
        try:
            from alpaca.data.live import StockDataStream
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        if self._data_stream is None:
            self._data_stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )

        async def _handler(bar):
            try:
                callback({
                    "symbol": bar.symbol,
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })
            except Exception as exc:  # noqa: BLE001
                logger.error("Bar callback error for %s: %s", bar.symbol, exc)

        self._data_stream.subscribe_bars(_handler, *symbols)
        logger.info("Subscribed to bars for: %s", symbols)

    def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to real-time NBBO quote updates."""
        try:
            from alpaca.data.live import StockDataStream
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        if self._data_stream is None:
            self._data_stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )

        async def _handler(quote):
            try:
                bid = float(quote.bid_price or 0)
                ask = float(quote.ask_price or 0)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
                callback({
                    "symbol": quote.symbol,
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": (ask - bid) / mid if mid > 0 else 0.0,
                    "timestamp": str(quote.timestamp),
                })
            except Exception as exc:  # noqa: BLE001
                logger.error("Quote callback error: %s", exc)

        self._data_stream.subscribe_quotes(_handler, *symbols)
        logger.info("Subscribed to quotes for: %s", symbols)

    def start_stream(self) -> None:
        """Start the WebSocket data stream in a daemon thread."""
        if self._data_stream is None:
            logger.warning("No subscriptions registered — call subscribe_bars() first")
            return
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return

        self._stream_thread = threading.Thread(
            target=self._data_stream.run,
            name="alpaca-data-stream",
            daemon=True,
        )
        self._stream_thread.start()
        logger.info("Data stream started")

    def stop_stream(self) -> None:
        """Stop the WebSocket data stream."""
        self._stop_data_stream()

    # ------------------------------------------------------------------
    # Streaming — trading updates (fills, order status)
    # ------------------------------------------------------------------

    def subscribe_trade_updates(
        self,
        callback: Callable[[dict], None],
    ) -> None:
        """
        Subscribe to order fill / cancel / status events.

        Parameters
        ----------
        callback : Callable[[dict], None]
            Called with a normalised update dict on each event.
        """
        try:
            from alpaca.trading.stream import TradingStream
        except ImportError as exc:
            raise ImportError("alpaca-py not installed") from exc

        if self._trading_stream is None:
            self._trading_stream = TradingStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper,
            )

        async def _handler(data):
            try:
                event = str(data.event)
                order = data.order
                update = {
                    "event": event,
                    "order_id": str(order.id),
                    "client_order_id": str(order.client_order_id),
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "type": str(order.order_type),
                    "status": str(order.status),
                    "qty": float(order.qty or 0),
                    "filled_qty": float(order.filled_qty or 0),
                    "filled_avg_price": float(order.filled_avg_price or 0),
                    "timestamp": str(data.timestamp) if hasattr(data, "timestamp") else None,
                }
                callback(update)
            except Exception as exc:  # noqa: BLE001
                logger.error("Trade update callback error: %s", exc)

        self._trading_stream.subscribe_trade_updates(_handler)

        t = threading.Thread(
            target=self._trading_stream.run,
            name="alpaca-trading-stream",
            daemon=True,
        )
        t.start()
        logger.info("Trading stream (trade updates) started")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_credentials(self) -> tuple[str, str]:
        """Read API key and secret from environment variables."""
        api_key = os.getenv("ALPACA_API_KEY", "").strip()
        secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "ALPACA_API_KEY not set. Add it to your .env file."
            )
        if not secret_key:
            raise EnvironmentError(
                "ALPACA_SECRET_KEY not set. Add it to your .env file."
            )
        return api_key, secret_key

    def _confirm_live_trading(self) -> None:
        """Interactive prompt that gates live-trading mode."""
        print("\n" + "=" * 60)
        print("⚠️  WARNING: YOU ARE ABOUT TO ENABLE LIVE TRADING")
        print("    Real money will be at risk.")
        print("=" * 60)
        try:
            response = input(
                f"Type '{_LIVE_CONFIRM_PHRASE}' to confirm: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("Live trading confirmation aborted.")
        if response != _LIVE_CONFIRM_PHRASE:
            raise RuntimeError(
                f"Live trading not confirmed (got '{response}'). "
                "Set ALPACA_PAPER=true or confirm correctly."
            )
        logger.warning("LIVE TRADING MODE CONFIRMED by user")

    def _retry_request(self, func: Callable, *args, max_retries: int = _MAX_RETRIES, **kwargs) -> Any:
        """
        Call `func(*args, **kwargs)` with exponential backoff on transient errors.

        Retries on HTTP 429, 5xx, and common connection errors.
        Raises the last exception if all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                is_retryable = (
                    status in _RETRYABLE_STATUS
                    or "timeout" in str(exc).lower()
                    or "connection" in str(exc).lower()
                )
                if not is_retryable or attempt == max_retries:
                    raise
                wait = min(30, _BACKOFF_BASE ** attempt)
                logger.warning(
                    "Alpaca API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    def _stop_data_stream(self) -> None:
        if self._data_stream is not None:
            try:
                self._data_stream.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error stopping data stream: %s", exc)
            self._data_stream = None

    def _stop_trading_stream(self) -> None:
        if self._trading_stream is not None:
            try:
                self._trading_stream.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error stopping trading stream: %s", exc)
            self._trading_stream = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timeframe(tf_str: str):
    """Convert string like '1Day', '1Hour', '5Min' to alpaca TimeFrame."""
    try:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError("alpaca-py not installed") from exc

    mapping = {
        "1min":  TimeFrame(1, TimeFrameUnit.Minute),
        "1minute": TimeFrame(1, TimeFrameUnit.Minute),
        "5min":  TimeFrame(5, TimeFrameUnit.Minute),
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
        "30min": TimeFrame(30, TimeFrameUnit.Minute),
        "1hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1day":  TimeFrame(1, TimeFrameUnit.Day),
        "1week": TimeFrame(1, TimeFrameUnit.Week),
        "1month": TimeFrame(1, TimeFrameUnit.Month),
    }
    return mapping.get(tf_str.lower().replace(" ", ""), TimeFrame(1, TimeFrameUnit.Day))
