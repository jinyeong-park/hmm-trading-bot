"""
backtest/backtester.py — Walk-forward allocation backtester.

DESIGN: Allocation-based, not trade-based.
Each bar, the system sets a TARGET PORTFOLIO ALLOCATION driven by the detected
volatility regime.  It rebalances only when the actual allocation drifts more
than `rebalance_threshold` from target.  No individual stop-losses — those are
live-trading-only constructs.

FILL DELAY: Signal generated at close of bar N → executed at open of bar N+1.
This prevents look-ahead from using today's close as both the signal and fill.

MARGIN / LEVERAGE: When leverage > 1.0 (e.g., 1.25x in low-vol), the position
value exceeds equity, making cash negative.
    equity = cash + Σ(shares × price)
    shares_value = equity × allocation × leverage
    cash = equity − shares_value  (negative when leveraged)
Alpaca supports 2x overnight leverage, so 1.25x is realistic.

MULTI-SYMBOL: One HMM is trained on the primary symbol's features (first key
of ohlcv_by_symbol, typically SPY).  The detected regime applies uniformly to
all symbols.  Equal weight allocation: target_weight_per_symbol = total_alloc / n.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config & result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Hyperparameters for a single backtest run."""

    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005       # one-way slippage fraction (5 bps)
    train_window: int = 252            # IS bars (≈ 1 trading year)
    test_window: int = 126             # OOS bars (≈ 6 months)
    step_size: int = 126               # walk-forward step (≈ 6 months)
    risk_free_rate: float = 0.045      # for Sharpe / Sortino
    commission_per_share: float = 0.0  # Alpaca is commission-free
    rebalance_threshold: float = 0.10  # minimum drift to trigger rebalance
    primary_symbol: Optional[str] = None  # HMM training symbol; None = first key


@dataclass
class BarRecord:
    """Single-bar snapshot stored in the OOS simulation loop."""

    date: pd.Timestamp
    equity: float
    cash: float
    regime: str
    regime_id: int
    regime_probability: float
    is_confirmed: bool
    allocation_total: float       # sum of all symbol weights × leverage
    n_rebalances: int             # rebalances executed on this bar


@dataclass
class TradeRecord:
    """Single rebalance event recorded as a trade."""

    signal_date: pd.Timestamp     # bar that triggered the rebalance
    exec_date: pd.Timestamp       # bar where order was filled (signal_date + 1)
    symbol: str
    side: str                     # 'buy' | 'sell'
    shares: float
    exec_price: float
    trade_value: float            # shares × exec_price (negative for sells)
    slippage_cost: float
    regime: str
    leverage: float
    portfolio_equity: float       # equity AFTER this fill


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_states: int                          # selected by BIC
    bic: float
    equity_curve: pd.Series               # DatetimeIndex → equity (float)
    bar_records: list[BarRecord]
    trades: pd.DataFrame                  # columns = TradeRecord fields
    regime_labels: pd.Series             # DatetimeIndex → regime label string
    confidence_history: pd.Series        # DatetimeIndex → float


@dataclass
class BacktestResult:
    """Aggregated result across all walk-forward folds."""

    config: BacktestConfig
    symbols: list[str] = field(default_factory=list)
    folds: list[FoldResult] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_labels: pd.Series = field(default_factory=pd.Series)
    confidence_history: pd.Series = field(default_factory=pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtester
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward allocation backtester.

    For each fold:
      1. Train a fresh HMMEngine on IS features (BIC model selection).
      2. Build a StrategyOrchestrator from the fitted model's regime_infos.
      3. Run the forward algorithm once on IS + OOS features (causal inference).
      4. Simulate the OOS period bar-by-bar with fill delay and slippage.

    Parameters
    ----------
    config : BacktestConfig
    hmm_config : dict
        kwargs forwarded to HMMEngine (e.g. n_candidates, n_init, …).
    feature_engineer : FeatureEngineer
        Pre-configured feature transformer.
    orchestrator_config : dict | None
        kwargs forwarded to StrategyOrchestrator (e.g. min_confidence, …).
    """

    def __init__(
        self,
        config: BacktestConfig,
        hmm_config: Optional[dict] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        orchestrator_config: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.hmm_config = hmm_config or {}
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.orchestrator_config = orchestrator_config or {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self, ohlcv_by_symbol: dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
            symbol → OHLCV DataFrame with DatetimeIndex.
            The first key is used as the primary symbol for HMM training.

        Returns
        -------
        BacktestResult
        """
        symbols = list(ohlcv_by_symbol.keys())
        primary = self.config.primary_symbol or symbols[0]
        if primary not in ohlcv_by_symbol:
            raise ValueError(f"primary_symbol '{primary}' not in ohlcv_by_symbol")

        logger.info(
            "WalkForwardBacktester.run: symbols=%s  primary=%s  capital=%.0f",
            symbols,
            primary,
            self.config.initial_capital,
        )

        # Compute features ONCE on the full primary OHLCV
        all_features = self.feature_engineer.transform(ohlcv_by_symbol[primary])
        logger.info("Features: %d valid bars (%s → %s)", len(all_features),
                    all_features.index[0].date(), all_features.index[-1].date())

        folds = self._generate_folds(all_features.index)
        if not folds:
            raise ValueError(
                f"No valid folds: need at least {self.config.train_window + self.config.test_window} "
                f"feature bars, got {len(all_features)}."
            )
        logger.info("Walk-forward folds: %d", len(folds))

        result = BacktestResult(config=self.config, symbols=symbols)
        current_equity = self.config.initial_capital

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(
                "Fold %d/%d  IS=%s→%s  OOS=%s→%s  equity=%.2f",
                fold_idx + 1, len(folds),
                train_idx[0].date(), train_idx[-1].date(),
                test_idx[0].date(), test_idx[-1].date(),
                current_equity,
            )
            fold_result = self._run_fold(
                fold_index=fold_idx,
                all_features=all_features,
                train_idx=train_idx,
                test_idx=test_idx,
                ohlcv_by_symbol=ohlcv_by_symbol,
                starting_equity=current_equity,
                symbols=symbols,
            )
            result.folds.append(fold_result)
            current_equity = float(fold_result.equity_curve.iloc[-1])

        result = self._stitch_folds(result)
        logger.info(
            "Backtest complete: final_equity=%.2f  total_trades=%d  folds=%d",
            float(result.equity_curve.iloc[-1]) if len(result.equity_curve) else 0,
            len(result.trades),
            len(result.folds),
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Fold generation
    # ─────────────────────────────────────────────────────────────────

    def _generate_folds(
        self, feature_index: pd.DatetimeIndex
    ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate (train_idx, test_idx) pairs for the walk-forward loop.

        Uses a ROLLING window of fixed train_window size; each fold advances by
        step_size bars.  The first fold's train window begins at bar 0.

        Parameters
        ----------
        feature_index : pd.DatetimeIndex
            Full chronological feature index (post warmup-drop).

        Returns
        -------
        list of (train DatetimeIndex, test DatetimeIndex) pairs
        """
        n = len(feature_index)
        tw = self.config.train_window
        ow = self.config.test_window
        step = self.config.step_size

        folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        start = 0

        while start + tw + ow <= n:
            train_end = start + tw
            test_end = train_end + ow
            folds.append((
                feature_index[start:train_end],
                feature_index[train_end:test_end],
            ))
            start += step

        return folds

    # ─────────────────────────────────────────────────────────────────
    # Single-fold simulation
    # ─────────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_index: int,
        all_features: pd.DataFrame,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        starting_equity: float,
        symbols: list[str],
    ) -> FoldResult:
        """
        Execute a single walk-forward fold.

        Steps:
          1. Train HMMEngine on IS features.
          2. Run forward algorithm on IS+OOS combined (causal).
          3. Build StrategyOrchestrator from fitted regime_infos.
          4. Simulate OOS bar-by-bar with fill delay.

        Parameters
        ----------
        fold_index : int
        all_features : pd.DataFrame
            Full feature matrix (entire dataset).
        train_idx : pd.DatetimeIndex
            IS feature dates.
        test_idx : pd.DatetimeIndex
            OOS feature dates.
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        starting_equity : float
        symbols : list[str]

        Returns
        -------
        FoldResult
        """
        # ── 1. Fit HMM on IS features ─────────────────────────────────
        train_features = all_features.loc[train_idx]
        hmm = HMMEngine(**self.hmm_config)
        try:
            hmm.fit(train_features)
        except Exception as exc:
            logger.error("Fold %d HMM fit failed: %s", fold_index, exc, exc_info=True)
            raise

        # ── 2. Run forward pass on IS + OOS (causal, no look-ahead) ───
        combined_idx = all_features.index[
            all_features.index.get_loc(train_idx[0]):
            all_features.index.get_loc(test_idx[-1]) + 1
        ]
        combined_features = all_features.loc[combined_idx]
        regime_states: list[RegimeState] = hmm.predict_regime_filtered(combined_features)

        # OOS regime states are the last len(test_idx) entries
        n_oos = len(test_idx)
        oos_regime_states = regime_states[-n_oos:]

        # ── 3. Build orchestrator from fitted regime infos ─────────────
        orch = StrategyOrchestrator(
            config={"strategy": self.orchestrator_config},
            regime_infos=hmm._regime_info,
        )

        # ── 4. Align OOS OHLCV data ────────────────────────────────────
        oos_ohlcv: dict[str, pd.DataFrame] = {}
        for sym, ohlcv in ohlcv_by_symbol.items():
            # Use all dates from train_start to test_end for bars context
            sym_dates = ohlcv.index.intersection(
                all_features.index[
                    all_features.index.get_loc(train_idx[0]):
                    all_features.index.get_loc(test_idx[-1]) + 1
                ]
            )
            oos_ohlcv[sym] = ohlcv.loc[sym_dates]

        # ── 5. OOS simulation loop ─────────────────────────────────────
        portfolio: dict = {
            "cash": starting_equity,
            "shares": {sym: 0.0 for sym in symbols},
        }
        pending_signals: dict[str, float] = {}   # symbol → target_weight

        equity_curve: dict[pd.Timestamp, float] = {}
        regime_history: dict[pd.Timestamp, str] = {}
        confidence_hist: dict[pd.Timestamp, float] = {}
        bar_records: list[BarRecord] = []
        all_trades: list[TradeRecord] = []

        n_symbols = len(symbols)

        for t, (oos_date, regime_state) in enumerate(zip(test_idx, oos_regime_states)):
            n_rebalances_today = 0

            # ── Execute pending signals at today's OPEN ──────────────
            if pending_signals:
                open_prices = self._get_open_prices(oos_ohlcv, oos_date, symbols)
                close_prices_prev = self._get_close_prices(oos_ohlcv, oos_date, symbols, offset=-1)
                eq_pre = self._compute_equity(portfolio, close_prices_prev)

                for sym, target_weight in pending_signals.items():
                    if open_prices.get(sym, 0) <= 0:
                        continue
                    exec_price = open_prices[sym]
                    current_shares = portfolio["shares"].get(sym, 0.0)
                    target_value = eq_pre * target_weight
                    target_shares = target_value / exec_price
                    delta = target_shares - current_shares

                    if abs(delta) < 1e-6:
                        continue

                    side = "buy" if delta > 0 else "sell"
                    exec_price_slip = self._apply_slippage(exec_price, side)
                    trade_val = delta * exec_price_slip
                    slip_cost = abs(delta) * abs(exec_price_slip - exec_price)

                    portfolio["cash"] -= trade_val
                    portfolio["shares"][sym] = target_shares

                    post_eq = self._compute_equity(
                        portfolio, self._get_close_prices(oos_ohlcv, oos_date, symbols)
                    )

                    signal_date = test_idx[t - 1] if t > 0 else oos_date
                    all_trades.append(TradeRecord(
                        signal_date=signal_date,
                        exec_date=oos_date,
                        symbol=sym,
                        side=side,
                        shares=abs(delta),
                        exec_price=exec_price_slip,
                        trade_value=abs(trade_val),
                        slippage_cost=slip_cost,
                        regime=regime_state.label.value,
                        leverage=1.0,
                        portfolio_equity=post_eq,
                    ))
                    n_rebalances_today += 1

                pending_signals = {}

            # ── Get today's close prices for equity & signal ─────────
            close_prices = self._get_close_prices(oos_ohlcv, oos_date, symbols)
            equity = self._compute_equity(portfolio, close_prices)

            # ── Build bars context for strategy (last 60 bars) ────────
            bars_context: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                if sym in oos_ohlcv:
                    sym_df = oos_ohlcv[sym]
                    loc = sym_df.index.searchsorted(oos_date, side="right")
                    bars_context[sym] = sym_df.iloc[max(0, loc - 60): loc]

            # ── Generate allocation signals ───────────────────────────
            signals = orch.generate_signals(
                symbols=symbols,
                bars_by_symbol=bars_context,
                regime_state=regime_state,
                is_flickering=hmm.is_flickering(),
            )

            # ── Check rebalance needed for each symbol ────────────────
            for sig in signals:
                # Equal-weight: split position_size_pct × leverage across symbols
                target_weight = sig.position_size_pct * sig.leverage / n_symbols
                sym = sig.symbol
                current_weight = (
                    portfolio["shares"].get(sym, 0.0) * close_prices.get(sym, 0.0) / equity
                    if equity > 0 else 0.0
                )
                if abs(current_weight - target_weight) > self.config.rebalance_threshold:
                    pending_signals[sym] = target_weight

            # ── Record equity & metadata ──────────────────────────────
            equity_curve[oos_date] = equity
            regime_history[oos_date] = regime_state.label.value
            confidence_hist[oos_date] = regime_state.probability

            total_alloc = sum(
                portfolio["shares"].get(sym, 0.0) * close_prices.get(sym, 0.0)
                for sym in symbols
            ) / equity if equity > 0 else 0.0

            bar_records.append(BarRecord(
                date=oos_date,
                equity=equity,
                cash=portfolio["cash"],
                regime=regime_state.label.value,
                regime_id=regime_state.state_id,
                regime_probability=regime_state.probability,
                is_confirmed=regime_state.is_confirmed,
                allocation_total=total_alloc,
                n_rebalances=n_rebalances_today,
            ))

        return FoldResult(
            fold_index=fold_index,
            train_start=train_idx[0],
            train_end=train_idx[-1],
            test_start=test_idx[0],
            test_end=test_idx[-1],
            n_states=hmm._n_states,
            bic=hmm._best_bic,
            equity_curve=pd.Series(equity_curve),
            bar_records=bar_records,
            trades=pd.DataFrame([vars(t) for t in all_trades]) if all_trades else pd.DataFrame(),
            regime_labels=pd.Series(regime_history),
            confidence_history=pd.Series(confidence_hist),
        )

    # ─────────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply one-way slippage.

        buy  → price × (1 + slippage_pct)   — fills slightly above market
        sell → price × (1 − slippage_pct)   — fills slightly below market
        """
        if side == "buy":
            return price * (1.0 + self.config.slippage_pct)
        return price * (1.0 - self.config.slippage_pct)

    @staticmethod
    def _compute_equity(
        portfolio: dict, close_prices: dict[str, float]
    ) -> float:
        """equity = cash + Σ(shares × close).  Cash may be negative (margin)."""
        equity = portfolio["cash"]
        for sym, shares in portfolio["shares"].items():
            equity += shares * close_prices.get(sym, 0.0)
        return equity

    @staticmethod
    def _get_close_prices(
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        symbols: list[str],
        offset: int = 0,
    ) -> dict[str, float]:
        """Return close prices on (date + offset) for each symbol."""
        prices: dict[str, float] = {}
        for sym in symbols:
            if sym not in ohlcv_by_symbol:
                continue
            df = ohlcv_by_symbol[sym]
            loc = df.index.searchsorted(date, side="right") - 1 + offset
            if 0 <= loc < len(df):
                prices[sym] = float(df["close"].iloc[loc])
        return prices

    @staticmethod
    def _get_open_prices(
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        symbols: list[str],
    ) -> dict[str, float]:
        """Return open prices on `date` for each symbol."""
        prices: dict[str, float] = {}
        for sym in symbols:
            if sym not in ohlcv_by_symbol:
                continue
            df = ohlcv_by_symbol[sym]
            if date in df.index:
                prices[sym] = float(df.loc[date, "open"])
        return prices

    @staticmethod
    def _stitch_folds(result: BacktestResult) -> BacktestResult:
        """Concatenate equity curves, trades, regime labels across all folds."""
        if not result.folds:
            return result

        equity_parts = [f.equity_curve for f in result.folds]
        result.equity_curve = pd.concat(equity_parts).sort_index()
        result.equity_curve = result.equity_curve[~result.equity_curve.index.duplicated(keep="last")]

        regime_parts = [f.regime_labels for f in result.folds]
        result.regime_labels = pd.concat(regime_parts).sort_index()
        result.regime_labels = result.regime_labels[~result.regime_labels.index.duplicated(keep="last")]

        conf_parts = [f.confidence_history for f in result.folds]
        result.confidence_history = pd.concat(conf_parts).sort_index()
        result.confidence_history = result.confidence_history[
            ~result.confidence_history.index.duplicated(keep="last")
        ]

        trade_parts = [f.trades for f in result.folds if len(f.trades) > 0]
        if trade_parts:
            result.trades = pd.concat(trade_parts, ignore_index=True)
            result.trades.sort_values("exec_date", inplace=True, ignore_index=True)
        else:
            result.trades = pd.DataFrame()

        return result
