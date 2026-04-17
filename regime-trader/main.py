"""
main.py — Entry point for regime-trader.

Usage
-----
  python main.py                          # paper trading (default)
  python main.py --dry-run                # full pipeline, no orders submitted
  python main.py --backtest               # walk-forward backtest
  python main.py --train-only             # train HMM and exit
  python main.py --stress-test            # run stress scenarios
  python main.py --compare                # benchmark comparisons
  python main.py --dashboard              # attach to running instance dashboard
  python main.py --live                   # live trading (requires confirmation)
  python main.py --symbols SPY QQQ NVDA  # override symbol list
  python main.py --config path/to.yaml   # custom config file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_STATE_FILE = _ROOT / "state_snapshot.json"
_MODEL_FILE = _ROOT / "models" / "hmm_model.pkl"
_LOG_DIR = _ROOT / "logs"
_MODEL_MAX_AGE_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup  (called early, before any imports that use logger)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO", log_dir: Path = _LOG_DIR) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            log_dir / "regime_trader.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(fh)
    except OSError:
        pass

    logging.basicConfig(level=numeric, format=fmt, datefmt=datefmt, handlers=handlers)


logger = logging.getLogger("regime_trader.main")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str = "config/settings.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = _ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.debug("Config loaded from %s", path)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="regime-trader",
        description="HMM-based volatility-regime allocation bot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live",        action="store_true", help="Live trading (real money — requires confirmation)")
    mode.add_argument("--dry-run",     action="store_true", help="Full pipeline, no orders submitted")
    mode.add_argument("--backtest",    action="store_true", help="Walk-forward backtest")
    mode.add_argument("--train-only",  action="store_true", help="Train HMM and exit")
    mode.add_argument("--stress-test", action="store_true", help="Run stress-test scenarios")
    mode.add_argument("--compare",     action="store_true", help="Benchmark comparisons")
    mode.add_argument("--dashboard",   action="store_true", help="Show dashboard for running instance")

    parser.add_argument("--config",  default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--symbols", nargs="*", default=None, help="Override symbol list (space-separated)")
    parser.add_argument("--output",  default="results/", help="Directory for backtest/stress output CSVs")
    parser.add_argument("--log-level", default=None, help="Override log level (DEBUG/INFO/WARNING/ERROR)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# State snapshot  (session recovery)
# ─────────────────────────────────────────────────────────────────────────────

def _save_state(state: dict, path: Path = _STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(path)
    logger.info("State snapshot saved → %s", path)


def _load_state(path: Path = _STATE_FILE) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        logger.info("Recovered state snapshot from %s", path)
        return state
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load state snapshot: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HMM model  (load or train)
# ─────────────────────────────────────────────────────────────────────────────

def _model_needs_retraining(model_path: Path, max_age_days: int = _MODEL_MAX_AGE_DAYS) -> bool:
    if not model_path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
    return age.days >= max_age_days


def _train_hmm(features: pd.DataFrame, cfg: dict, model_path: Path):
    """Fit HMMEngine and save to disk.  Returns fitted engine."""
    from core.hmm_engine import HMMEngine

    hmm_cfg = cfg.get("hmm", {})
    engine = HMMEngine(
        n_candidates=hmm_cfg.get("n_candidates", [3, 4, 5, 6, 7]),
        n_init=hmm_cfg.get("n_init", 10),
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        min_train_bars=hmm_cfg.get("min_train_bars", 252),
        stability_bars=hmm_cfg.get("stability_bars", 3),
        min_confidence=hmm_cfg.get("min_confidence", 0.55),
    )
    logger.info("Training HMM on %d bars ...", len(features))
    engine.fit(features)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(str(model_path))
    logger.info("HMM trained (n_states=%s) → %s", engine.n_states, model_path)
    return engine


def _load_or_train_hmm(features: pd.DataFrame, cfg: dict, model_path: Path = _MODEL_FILE):
    """Load a cached model if fresh; otherwise retrain."""
    from core.hmm_engine import HMMEngine

    if not _model_needs_retraining(model_path):
        try:
            engine = HMMEngine.load(str(model_path))
            age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            logger.info("Loaded HMM model (age=%dd, n_states=%s)", age.days, engine.n_states)
            return engine
        except Exception as exc:
            logger.warning("Failed to load cached model (%s) — retraining", exc)

    return _train_hmm(features, cfg, model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Session summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_session_summary(session_start: datetime, trades_submitted: int, signals_rejected: int, regime_counts: dict) -> None:
    duration = datetime.now(tz=timezone.utc) - session_start
    hours, rem = divmod(int(duration.total_seconds()), 3600)
    minutes = rem // 60
    print("\n" + "=" * 56)
    print("  SESSION SUMMARY")
    print("=" * 56)
    print(f"  Duration         : {hours}h {minutes}m")
    print(f"  Trades submitted : {trades_submitted}")
    print(f"  Signals rejected : {signals_rejected}")
    if regime_counts:
        print("  Regime breakdown :")
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {regime:<18} {count:>4} bars")
    print("=" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# Live / paper trading loop
# ─────────────────────────────────────────────────────────────────────────────

class TradingSession:
    """
    Orchestrates the full live/paper trading loop.

    One instance is created per `python main.py` invocation.
    Handles startup, the main bar-event loop, graceful shutdown,
    and unhandled-exception recovery.
    """

    def __init__(self, cfg: dict, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run

        broker_cfg   = cfg.get("broker", {})
        hmm_cfg      = cfg.get("hmm", {})
        strategy_cfg = cfg.get("strategy", {})
        risk_cfg     = cfg.get("risk", {})

        self.symbols: list[str] = broker_cfg.get("symbols", ["SPY"])
        self.primary_symbol: str = self.symbols[0]
        self.paper: bool = broker_cfg.get("paper_trading", True)
        self.timeframe: str = broker_cfg.get("timeframe", "1Day")

        # Session stats
        self._session_start = datetime.now(tz=timezone.utc)
        self._trades_submitted = 0
        self._signals_rejected = 0
        self._regime_counts: dict[str, int] = {}
        self._last_regime_label: str = "UNKNOWN"
        self._last_weekly_retrain: Optional[datetime] = None

        # Data feed state
        self._bar_buffer: dict[str, list] = {s: [] for s in self.symbols}
        self._feed_healthy: bool = True
        self._feed_dropped_at: Optional[datetime] = None
        self._shutdown_event = threading.Event()

        # Components — initialised in startup()
        self._client = None
        self._data_feed = None
        self._feature_engineer = None
        self._hmm_engine = None
        self._orchestrator = None
        self._risk_manager = None
        self._position_tracker = None
        self._order_executor = None

        # Per-symbol rolling bar cache (DatetimeIndex OHLCV DataFrames)
        self._bar_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Full startup sequence. Raises on unrecoverable error."""
        _print_banner(self.dry_run, self.paper)

        # ── 1. Load config & connect to Alpaca ───────────────────────
        logger.info("[1/8] Connecting to Alpaca (%s mode) ...", "paper" if self.paper else "LIVE")
        from broker.alpaca_client import AlpacaClient
        self._client = AlpacaClient(paper=self.paper)
        self._client.connect()

        acc = self._client.get_account()
        logger.info(
            "Account: status=%s  equity=$%.2f  buying_power=$%.2f",
            acc["status"], acc["equity"], acc["buying_power"],
        )
        if acc.get("trading_blocked"):
            raise RuntimeError("Account is trading-blocked — check Alpaca dashboard")

        # ── 2. Market hours ──────────────────────────────────────────
        logger.info("[2/8] Checking market hours ...")
        clock = self._client.get_clock()
        if not clock["is_open"]:
            next_open = clock["next_open"]
            logger.info("Market closed.  Next open: %s", next_open)
            _wait_for_market_open(self._client, self._shutdown_event)
            if self._shutdown_event.is_set():
                return

        # ── 3. Initialize feature engineer & fetch historical bars ───
        logger.info("[3/8] Fetching historical data ...")
        from data.feature_engineering import FeatureEngineer
        from data.market_data import MarketDataFeed

        hmm_cfg = self.cfg.get("hmm", {})
        self._feature_engineer = FeatureEngineer(
            vol_window=20,
            sma_slow=200,
            normalise=True,
        )

        self._data_feed = MarketDataFeed(
            client=self._client,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback_bars=600,
        )
        self._data_feed.initialize()

        # Build bar cache and compute features on primary symbol
        for sym in self.symbols:
            self._bar_cache[sym] = self._data_feed.get_bars(sym)

        primary_bars = self._bar_cache.get(self.primary_symbol, pd.DataFrame())
        if len(primary_bars) < hmm_cfg.get("min_train_bars", 252):
            raise RuntimeError(
                f"Insufficient history for {self.primary_symbol}: "
                f"{len(primary_bars)} bars < {hmm_cfg.get('min_train_bars', 252)}"
            )

        # ── 4. Load or train HMM ─────────────────────────────────────
        logger.info("[4/8] Loading / training HMM ...")
        features = self._feature_engineer.transform(primary_bars)
        self._hmm_engine = _load_or_train_hmm(features, self.cfg)

        # ── 5. Initialize risk manager ───────────────────────────────
        logger.info("[5/8] Initializing risk manager ...")
        from core.risk_manager import RiskManager
        risk_cfg = self.cfg.get("risk", {})
        self._risk_manager = RiskManager(
            max_risk_per_trade=risk_cfg.get("max_risk_per_trade", 0.01),
            max_exposure=risk_cfg.get("max_exposure", 0.80),
            max_leverage=risk_cfg.get("max_leverage", 1.25),
            max_single_position=risk_cfg.get("max_single_position", 0.15),
            max_concurrent=risk_cfg.get("max_concurrent", 5),
            max_daily_trades=risk_cfg.get("max_daily_trades", 20),
            daily_dd_reduce=risk_cfg.get("daily_dd_reduce", 0.02),
            daily_dd_halt=risk_cfg.get("daily_dd_halt", 0.03),
            weekly_dd_reduce=risk_cfg.get("weekly_dd_reduce", 0.05),
            weekly_dd_halt=risk_cfg.get("weekly_dd_halt", 0.07),
            max_dd_from_peak=risk_cfg.get("max_dd_from_peak", 0.10),
        )
        self._risk_manager.update_equity(acc["equity"], datetime.now(tz=timezone.utc))

        # ── 6. Initialize position tracker ───────────────────────────
        logger.info("[6/8] Syncing positions ...")
        from broker.position_tracker import PositionTracker
        self._position_tracker = PositionTracker(
            client=self._client,
            circuit_breaker=self._risk_manager.circuit_breaker,
        )
        self._position_tracker.start()

        # ── 7. Initialize strategy orchestrator ──────────────────────
        logger.info("[7/8] Initializing strategy orchestrator ...")
        from core.regime_strategies import StrategyOrchestrator
        strategy_cfg = self.cfg.get("strategy", {})
        self._orchestrator = StrategyOrchestrator(
            low_vol_allocation=strategy_cfg.get("low_vol_allocation", 0.95),
            mid_vol_allocation_trend=strategy_cfg.get("mid_vol_allocation_trend", 0.95),
            mid_vol_allocation_no_trend=strategy_cfg.get("mid_vol_allocation_no_trend", 0.60),
            high_vol_allocation=strategy_cfg.get("high_vol_allocation", 0.60),
            uncertainty_size_mult=strategy_cfg.get("uncertainty_size_mult", 0.50),
            rebalance_threshold=strategy_cfg.get("rebalance_threshold", 0.10),
        )

        # ── 8. Order executor ─────────────────────────────────────────
        from broker.order_executor import OrderExecutor
        self._order_executor = OrderExecutor(
            client=self._client,
            use_limit_orders=True,
            on_fill=self._on_fill,
        )

        # ── 9. Recover session state if available ────────────────────
        prior_state = _load_state()
        if prior_state:
            logger.info("Recovered prior session state (regime=%s)", prior_state.get("last_regime"))
            self._last_regime_label = prior_state.get("last_regime", "UNKNOWN")

        # ── 10. Start data stream ─────────────────────────────────────
        logger.info("[8/8] Starting WebSocket data feed ...")
        self._data_feed.subscribe_bars(self.symbols, self.timeframe, self._on_bar)
        self._data_feed.start_stream()

        _print_system_state(acc, self.symbols, self._last_regime_label, self.dry_run)
        logger.info("System online ✓")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Block until shutdown.  Bar events arrive via _on_bar() WebSocket callback
        and are processed on the main thread via a thread-safe queue.
        """
        import queue as _queue
        self._bar_queue: _queue.Queue = _queue.Queue()

        def _drain():
            while not self._shutdown_event.is_set():
                try:
                    sym, bar = self._bar_queue.get(timeout=1.0)
                    self._process_bar(sym, bar)
                except _queue.Empty:
                    pass
                except Exception as exc:
                    logger.error("Unhandled error in bar processing: %s", exc)
                    logger.debug(traceback.format_exc())
                    self._save_session_state()

        drain_thread = threading.Thread(target=_drain, name="bar-drain", daemon=True)
        drain_thread.start()

        logger.info("Main loop running. Press Ctrl+C to stop.")
        try:
            while not self._shutdown_event.is_set():
                time.sleep(5)
                self._order_executor.sync_order_statuses()
        except KeyboardInterrupt:
            pass

    def _on_bar(self, symbol: str, bar: pd.Series) -> None:
        """WebSocket callback — enqueue bar for main-thread processing."""
        if not self._feed_healthy:
            self._feed_healthy = True
            self._feed_dropped_at = None
            logger.info("Data feed restored for %s", symbol)
        try:
            self._bar_queue.put_nowait((symbol, bar))
        except Exception:
            pass  # queue not yet created during startup

    def _process_bar(self, symbol: str, bar: pd.Series) -> None:
        """
        Full per-bar pipeline.
        Steps 1-11 from the spec, executed on each incoming bar.
        """
        ts = bar.name if hasattr(bar, "name") else datetime.now(tz=timezone.utc)
        logger.debug("Processing bar: %s @ %s", symbol, ts)

        # ── 1. Update bar cache ──────────────────────────────────────
        self._append_bar(symbol, bar)

        # Only drive the full pipeline on the primary symbol bar
        # (secondary symbols update their caches but don't trigger signals)
        if symbol != self.primary_symbol:
            return

        # ── 2. Compute features (causal — no future data) ────────────
        primary_bars = self._bar_cache.get(self.primary_symbol)
        if primary_bars is None or len(primary_bars) < 252:
            logger.debug("Insufficient bars for feature computation (%d)", len(primary_bars) if primary_bars is not None else 0)
            return

        try:
            features = self._feature_engineer.transform(primary_bars)
        except Exception as exc:
            logger.warning("Feature computation failed: %s", exc)
            return

        if len(features) < 50:
            return

        # ── 3. HMM forward-algorithm prediction ──────────────────────
        try:
            regime_states = self._hmm_engine.predict_regime_filtered(features)
            if not regime_states:
                return
            regime_state = regime_states[-1]
        except Exception as exc:
            logger.warning("HMM prediction failed — holding current regime: %s", exc)
            return  # hold current regime, keep stops active

        # ── 4 & 5. Stability + flicker ───────────────────────────────
        regime_label = str(regime_state.label)
        is_confirmed = regime_state.is_confirmed
        consecutive = regime_state.consecutive_bars

        # Count flicker: transitions in last flicker_window bars
        flicker_window = self.cfg.get("hmm", {}).get("flicker_window", 20)
        flicker_threshold = self.cfg.get("hmm", {}).get("flicker_threshold", 4)
        is_flickering = (
            not is_confirmed
            or consecutive < self.cfg.get("hmm", {}).get("stability_bars", 3)
        )

        self._regime_counts[regime_label] = self._regime_counts.get(regime_label, 0) + 1
        self._last_regime_label = regime_label
        self._position_tracker.set_current_regime(regime_label)

        # ── 6. Strategy orchestrator → signals ───────────────────────
        bars_by_symbol: dict[str, pd.DataFrame] = {
            s: self._bar_cache[s] for s in self.symbols if s in self._bar_cache
        }
        try:
            signals = self._orchestrator.generate_signals(
                symbols=self.symbols,
                bars_by_symbol=bars_by_symbol,
                regime_state=regime_state,
                is_flickering=is_flickering,
            )
        except Exception as exc:
            logger.warning("Signal generation failed: %s", exc)
            return

        # ── 7. Risk validation → order submission ────────────────────
        snap = self._position_tracker.get_snapshot()
        self._risk_manager.update_equity(snap.equity, ts if isinstance(ts, datetime) else datetime.now(tz=timezone.utc))

        portfolio_state = self._position_tracker.get_portfolio_state(
            flicker_rate=1.0 if is_flickering else 0.0,
        )

        for sig in signals:
            self._handle_signal(sig, portfolio_state, is_overnight=False)

        # ── 8. Update trailing stops ─────────────────────────────────
        self._update_trailing_stops(regime_label, bars_by_symbol)

        # ── 9. Circuit breaker check ─────────────────────────────────
        cb = self._risk_manager.circuit_breaker.check(portfolio_state, hmm_regime=regime_label)
        if cb.action.value == "halt":
            logger.warning("CIRCUIT BREAKER ACTIVE: %s", cb.reason)

        # ── 10. Dashboard refresh ────────────────────────────────────
        self._refresh_dashboard(regime_state, snap)

        # ── 11. Weekly HMM retrain ───────────────────────────────────
        now = datetime.now(tz=timezone.utc)
        if (
            self._last_weekly_retrain is None
            or (now - self._last_weekly_retrain).days >= 7
        ):
            self._schedule_retrain(features)
            self._last_weekly_retrain = now

    def _handle_signal(self, sig, portfolio_state, is_overnight: bool) -> None:
        """Validate a single signal through the risk manager and submit if approved."""
        decision = self._risk_manager.validate_signal(
            sig, portfolio_state, is_overnight=is_overnight
        )

        sym = sig.symbol
        direction = str(getattr(sig, "direction", "LONG")).upper()

        if decision.approved:
            if decision.modifications:
                mod_summary = "; ".join(f"{m.field}: {m.original}→{m.modified}" for m in decision.modifications)
                logger.info("Signal MODIFIED [%s %s]: %s", sym, direction, mod_summary)

            if self.dry_run:
                logger.info("[DRY-RUN] Would submit: %s %s size=%.4f lev=%.2f",
                            sym, direction,
                            getattr(decision.modified_signal, "position_size_pct", 0),
                            getattr(decision.modified_signal, "leverage", 1.0))
                return

            latest = self._client.get_latest_bar(sym)
            if latest is None:
                logger.warning("No price available for %s — signal skipped", sym)
                return

            price = latest["close"]
            equity = portfolio_state.equity
            entry = getattr(decision.modified_signal, "entry_price", price) or price
            stop  = getattr(decision.modified_signal, "stop_loss", None)
            if stop is None:
                logger.warning("Signal for %s has no stop_loss — skipped", sym)
                return

            qty = self._risk_manager.compute_position_size(
                equity=equity,
                entry_price=float(entry),
                stop_price=float(stop),
                allocation_fraction=getattr(decision.modified_signal, "position_size_pct", 0.10),
                is_overnight=is_overnight,
            )

            take_profit = getattr(decision.modified_signal, "take_profit", None)
            if take_profit is not None:
                order = self._order_executor.submit_bracket_order(
                    decision.modified_signal, price, round(qty, 0)
                )
            else:
                order = self._order_executor.submit_order(
                    decision.modified_signal, price, round(qty, 0)
                )

            if order:
                self._trades_submitted += 1
                self._risk_manager.register_trade()
                logger.info(
                    "Order submitted: %s %s qty=%.0f entry=%.4f stop=%.4f trade_id=%s",
                    sym, direction, qty, entry, float(stop), order.trade_id,
                )

        else:
            self._signals_rejected += 1
            logger.info("Signal REJECTED [%s %s]: %s", sym, direction, decision.rejection_reason)

    # ------------------------------------------------------------------
    # Trailing stops
    # ------------------------------------------------------------------

    def _update_trailing_stops(self, regime_label: str, bars_by_symbol: dict) -> None:
        """
        Tighten trailing stops based on current regime.
        - LOW_VOL: trail at EMA50 - 0.5 ATR
        - MID_VOL: trail at EMA50 - 0.5 ATR
        - HIGH_VOL: trail at EMA50 - 1.0 ATR  (wider — avoid whipsaw)
        """
        positions = self._position_tracker.get_all_positions()
        if not positions:
            return

        atr_mult = 1.0 if "HIGH" in regime_label.upper() else 0.5

        for sym, pos in positions.items():
            bars = bars_by_symbol.get(sym)
            if bars is None or len(bars) < 50:
                continue
            close = bars["close"]
            high  = bars["high"]
            low   = bars["low"]
            prev_close = close.shift(1)

            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

            new_stop = round(ema50 - atr_mult * atr, 4)

            if pos.side == "long" and new_stop > pos.stop_price:
                if not self.dry_run:
                    self._order_executor.modify_stop(sym, new_stop)
                    logger.info("Trailing stop tightened: %s %.4f → %.4f", sym, pos.stop_price, new_stop)

    # ------------------------------------------------------------------
    # Fill callback
    # ------------------------------------------------------------------

    def _on_fill(self, order) -> None:
        """Called by OrderExecutor when a fill is confirmed."""
        logger.info(
            "FILL: %s %s qty=%.2f @ $%.4f  trade_id=%s",
            order.side.value.upper(), order.symbol,
            order.filled_qty, order.filled_avg_price or 0,
            order.trade_id,
        )

    # ------------------------------------------------------------------
    # Weekly retrain
    # ------------------------------------------------------------------

    def _schedule_retrain(self, current_features: pd.DataFrame) -> None:
        """Retrain HMM in a background thread (non-blocking)."""
        def _retrain():
            logger.info("Weekly HMM retrain starting ...")
            try:
                new_engine = _train_hmm(current_features, self.cfg, _MODEL_FILE)
                self._hmm_engine = new_engine
                logger.info("Weekly HMM retrain complete (n_states=%s)", new_engine.n_states)
            except Exception as exc:
                logger.error("Weekly retrain failed — keeping current model: %s", exc)

        t = threading.Thread(target=_retrain, name="hmm-retrain", daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _refresh_dashboard(self, regime_state, snap) -> None:
        """Print a compact one-line status to stdout."""
        regime = str(regime_state.label)
        conf   = getattr(regime_state, "probability", getattr(regime_state, "confidence", 0))
        eq     = snap.equity
        n_pos  = len(snap.positions)
        tag    = "[DRY]" if self.dry_run else ""
        logger.debug(
            "%s Regime=%-12s conf=%.0f%%  equity=$%,.0f  positions=%d",
            tag, regime, conf * 100, eq, n_pos,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Graceful shutdown:
          - Signal the main loop to stop.
          - Close WebSocket connections.
          - Do NOT close open positions (stops remain active).
          - Save state snapshot.
          - Print session summary.
        """
        logger.info("Shutdown initiated ...")
        self._shutdown_event.set()

        if self._data_feed is not None:
            try:
                self._data_feed.stop_stream()
            except Exception as exc:
                logger.debug("Error stopping data feed: %s", exc)

        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as exc:
                logger.debug("Error disconnecting client: %s", exc)

        self._save_session_state()
        _print_session_summary(
            self._session_start,
            self._trades_submitted,
            self._signals_rejected,
            self._regime_counts,
        )
        logger.info("Shutdown complete.")

    def _save_session_state(self) -> None:
        state = {
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "last_regime": self._last_regime_label,
            "regime_counts": self._regime_counts,
            "trades_submitted": self._trades_submitted,
            "signals_rejected": self._signals_rejected,
            "symbols": self.symbols,
        }
        _save_state(state)

    # ------------------------------------------------------------------
    # Bar cache helper
    # ------------------------------------------------------------------

    def _append_bar(self, symbol: str, bar: pd.Series) -> None:
        existing = self._bar_cache.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        new_row = pd.DataFrame(
            [[bar.get("open", bar.iloc[0] if len(bar) > 0 else 0),
              bar.get("high", bar.iloc[1] if len(bar) > 1 else 0),
              bar.get("low",  bar.iloc[2] if len(bar) > 2 else 0),
              bar.get("close", bar.iloc[3] if len(bar) > 3 else 0),
              bar.get("volume", bar.iloc[4] if len(bar) > 4 else 0)]],
            columns=["open", "high", "low", "close", "volume"],
            index=[bar.name if hasattr(bar, "name") else pd.Timestamp.utcnow()],
        )
        updated = pd.concat([existing, new_row]).sort_index()
        # Drop duplicates by index
        updated = updated[~updated.index.duplicated(keep="last")]
        # Keep rolling window
        if len(updated) > 600:
            updated = updated.iloc[-600:]
        self._bar_cache[symbol] = updated


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_live(cfg: dict, dry_run: bool = False) -> None:
    """Start the live/paper trading session."""
    session = TradingSession(cfg, dry_run=dry_run)

    # Register shutdown handlers
    def _sig_handler(signum, frame):
        logger.info("Signal %d received — shutting down ...", signum)
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        session.startup()
        session.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc)
        logger.debug(traceback.format_exc())
        try:
            session._save_session_state()
        except Exception:
            pass
        raise
    finally:
        session.shutdown()


def run_backtest(cfg: dict, output_dir: str = "results/") -> None:
    """Walk-forward backtest → performance report → CSV exports."""
    from dotenv import load_dotenv
    load_dotenv()

    from broker.alpaca_client import AlpacaClient
    from data.feature_engineering import FeatureEngineer
    from backtest.backtester import BacktestConfig, WalkForwardBacktester
    from backtest.performance import PerformanceAnalyser

    broker_cfg  = cfg.get("broker", {})
    backtest_cfg = cfg.get("backtest", {})
    symbols = broker_cfg.get("symbols", ["SPY"])

    logger.info("Fetching historical data for backtest (%s) ...", symbols)
    client = AlpacaClient(paper=True)
    client.connect()

    end   = pd.Timestamp.utcnow()
    start = end - timedelta(days=365 * 5)   # 5 years
    bars_by_symbol = client.get_bars(symbols, "1Day", start, end)

    bt_cfg = BacktestConfig(
        initial_capital=backtest_cfg.get("initial_capital", 100_000),
        slippage_pct=backtest_cfg.get("slippage_pct", 0.0005),
        train_window=backtest_cfg.get("train_window", 252),
        test_window=backtest_cfg.get("test_window", 126),
        step_size=backtest_cfg.get("step_size", 126),
        risk_free_rate=backtest_cfg.get("risk_free_rate", 0.045),
        primary_symbol=symbols[0],
    )
    backtester = WalkForwardBacktester(bt_cfg, cfg)
    logger.info("Running walk-forward backtest ...")
    result = backtester.run(bars_by_symbol)

    analyser = PerformanceAnalyser(risk_free_rate=bt_cfg.risk_free_rate)
    benchmark_ohlcv = bars_by_symbol.get(symbols[0])
    report = analyser.analyse(result, benchmark_ohlcv=benchmark_ohlcv)

    analyser.print_report(report)
    paths = analyser.export_csv(result, report, output_dir=output_dir)
    logger.info("Results written to: %s", {k: str(v) for k, v in paths.items()})
    client.disconnect()


def run_train_only(cfg: dict) -> None:
    """Fetch data, train HMM, save model, exit."""
    from dotenv import load_dotenv
    load_dotenv()

    from broker.alpaca_client import AlpacaClient
    from data.feature_engineering import FeatureEngineer

    broker_cfg = cfg.get("broker", {})
    symbols    = broker_cfg.get("symbols", ["SPY"])
    primary    = symbols[0]

    client = AlpacaClient(paper=True)
    client.connect()

    end   = pd.Timestamp.utcnow()
    start = end - timedelta(days=365 * 3)
    bars  = client.get_bars([primary], "1Day", start, end)[primary]
    client.disconnect()

    engineer = FeatureEngineer(normalise=True)
    features = engineer.transform(bars)
    engine = _train_hmm(features, cfg, _MODEL_FILE)

    print(f"\nHMM trained: n_states={engine.n_states}  features={len(features)} bars")
    print(f"Model saved → {_MODEL_FILE}")


def run_stress_test(cfg: dict, output_dir: str = "results/") -> None:
    """Run stress scenarios and print summary."""
    from dotenv import load_dotenv
    load_dotenv()

    from broker.alpaca_client import AlpacaClient
    from backtest.backtester import BacktestConfig, WalkForwardBacktester
    from backtest.stress_test import StressTester, CrashScenario, GapScenario

    broker_cfg  = cfg.get("broker", {})
    backtest_cfg = cfg.get("backtest", {})
    symbols = broker_cfg.get("symbols", ["SPY"])

    client = AlpacaClient(paper=True)
    client.connect()
    end   = pd.Timestamp.utcnow()
    start = end - timedelta(days=365 * 3)
    bars_by_symbol = client.get_bars(symbols, "1Day", start, end)
    client.disconnect()

    bt_cfg = BacktestConfig(
        initial_capital=backtest_cfg.get("initial_capital", 100_000),
        primary_symbol=symbols[0],
    )
    backtester = WalkForwardBacktester(bt_cfg, cfg)
    tester = StressTester(backtester)

    # Monte Carlo: 100 seeds, 10 crash points per sim, -5% to -15% gaps
    print("\n=== Monte Carlo Crash Test (n=100) ===")
    mc = tester.run_monte_carlo_crashes(bars_by_symbol, n_simulations=100)
    print(f"  Crash range      : -{mc.crash_pct_min:.0f}% to -{mc.crash_pct_max:.0f}%")
    print(f"  Final equity mean: ${mc.final_equity_mean:,.0f}")
    print(f"  Final equity p5  : ${mc.final_equity_p5:,.0f}  (5th percentile)")
    print(f"  Max DD mean      : {mc.max_drawdown_mean:.1f}%")
    print(f"  Max DD p95       : {mc.max_drawdown_p95:.1f}%  (95th percentile)")
    print(f"  Survived (>$0)   : {mc.pct_survived:.0f}%")

    # Gap risk: 20 random overnight gaps 2-5× ATR
    print("\n=== Gap Risk Test (n=20 random gaps) ===")
    gap_results = tester.run_gap_risk(bars_by_symbol, n_gaps=20)
    returns = [r.total_return_pct for r in gap_results]
    mdds    = [r.max_drawdown_pct for r in gap_results]
    print(f"  Return range  : {min(returns):.1f}% to {max(returns):.1f}%")
    print(f"  Max DD range  : {min(mdds):.1f}% to {max(mdds):.1f}%")

    # Regime misclassification: 10 trials
    print("\n=== Regime Misclassification Test (n=10, shuffle=20%) ===")
    misc = tester.run_regime_misclassification(bars_by_symbol, n_trials=10)
    if misc:
        misc_returns = [r.total_return_pct for r in misc]
        misc_mdds    = [r.max_drawdown_pct for r in misc]
        print(f"  Return range  : {min(misc_returns):.1f}% to {max(misc_returns):.1f}%")
        print(f"  Max DD range  : {min(misc_mdds):.1f}% to {max(misc_mdds):.1f}%")


def run_compare(cfg: dict, output_dir: str = "results/") -> None:
    """Full backtest + benchmark comparison table."""
    run_backtest(cfg, output_dir=output_dir)


def run_dashboard(_cfg: dict) -> None:
    """Attach to a running instance by tailing the log file."""
    log_file = _LOG_DIR / "regime_trader.log"
    if not log_file.exists():
        print(f"No log file found at {log_file} — is the bot running?")
        sys.exit(1)
    print(f"Tailing {log_file} (Ctrl+C to exit) ...\n")
    import subprocess
    try:
        subprocess.run(["tail", "-f", "-n", "50", str(log_file)])
    except KeyboardInterrupt:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_market_open(client, shutdown_event: threading.Event) -> None:
    """Sleep until market opens, checking every 60 seconds."""
    while not shutdown_event.is_set():
        clock = client.get_clock()
        if clock["is_open"]:
            return
        next_open = pd.Timestamp(clock["next_open"])
        wait_sec  = max(0, (next_open - pd.Timestamp.utcnow().tz_localize("UTC")).total_seconds())
        if wait_sec > 0:
            logger.info("Market closed — waiting %.0f min until %s ...", wait_sec / 60, next_open)
            # Sleep in 60s chunks so SIGINT is responsive
            for _ in range(min(60, int(wait_sec))):
                if shutdown_event.is_set():
                    return
                time.sleep(1)
        else:
            return


def _print_banner(dry_run: bool, paper: bool) -> None:
    mode = "DRY-RUN" if dry_run else ("PAPER" if paper else "⚠️  LIVE")
    print("\n" + "=" * 56)
    print(f"  regime-trader  [{mode}]")
    print("=" * 56)


def _print_system_state(acc: dict, symbols: list[str], regime: str, dry_run: bool) -> None:
    print(f"\n{'─'*56}")
    print(f"  Equity        : ${acc['equity']:>12,.2f}")
    print(f"  Cash          : ${acc['cash']:>12,.2f}")
    print(f"  Buying Power  : ${acc['buying_power']:>12,.2f}")
    print(f"  Symbols       : {', '.join(symbols)}")
    print(f"  Last regime   : {regime}")
    print(f"  Dry-run       : {dry_run}")
    print(f"{'─'*56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Bootstrap env, config, logging, then dispatch to the chosen mode."""
    load_dotenv()
    args = _parse_args()

    # Config
    try:
        cfg = _load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Logging
    log_cfg = cfg.get("monitoring", {})
    log_level = args.log_level or log_cfg.get("log_level", "INFO")
    _setup_logging(log_level, log_dir=_ROOT / log_cfg.get("log_dir", "logs/"))

    # Symbol override
    if args.symbols:
        cfg.setdefault("broker", {})["symbols"] = args.symbols

    # Force paper unless --live
    if not args.live:
        cfg.setdefault("broker", {})["paper_trading"] = True

    output = args.output

    # Dispatch
    if args.backtest:
        run_backtest(cfg, output_dir=output)
    elif args.train_only:
        run_train_only(cfg)
    elif args.stress_test:
        run_stress_test(cfg, output_dir=output)
    elif args.compare:
        run_compare(cfg, output_dir=output)
    elif args.dashboard:
        run_dashboard(cfg)
    else:
        # Default: paper (or --live if flag set)
        run_live(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
