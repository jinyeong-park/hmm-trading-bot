"""
backtester.py — Walk-forward allocation backtester.

Responsibilities:
  - Implement a walk-forward loop: train HMM on window, test on out-of-sample period.
  - Simulate order execution with configurable slippage.
  - Track equity curve, position history, and regime labels across all folds.
  - Return a BacktestResult consumed by performance.py for metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import RegimeStrategy
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from data.feature_engineering import FeatureEngineer


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005
    train_window: int = 252
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045
    commission_per_share: float = 0.0   # Set > 0 for per-share commissions


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    equity_curve: pd.Series               # DatetimeIndex → equity value
    trades: pd.DataFrame                  # Trade log for this fold
    regime_labels: pd.Series             # DatetimeIndex → regime string


@dataclass
class BacktestResult:
    """Aggregated result across all walk-forward folds."""

    config: BacktestConfig
    folds: list[FoldResult] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_labels: pd.Series = field(default_factory=pd.Series)


class WalkForwardBacktester:
    """
    Runs a walk-forward backtest over a full historical dataset.

    Parameters
    ----------
    config : BacktestConfig
        Backtest hyperparameters.
    hmm_engine : HMMEngine
        HMM model instance (re-fit on each fold).
    strategy : RegimeStrategy
        Allocation strategy applied to regime signals.
    risk_manager : RiskManager
        Risk layer (re-initialised for each fold).
    feature_engineer : FeatureEngineer
        Transforms OHLCV into the HMM feature matrix.
    """

    def __init__(
        self,
        config: BacktestConfig,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
        feature_engineer: FeatureEngineer,
    ) -> None:
        self.config = config
        self.hmm_engine = hmm_engine
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.feature_engineer = feature_engineer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
            Mapping symbol → OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        BacktestResult
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_folds(
        self, index: pd.DatetimeIndex
    ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate (train_index, test_index) pairs for the walk-forward loop.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Full chronological date index of the dataset.

        Returns
        -------
        list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        """
        raise NotImplementedError

    def _run_fold(
        self,
        fold_index: int,
        train_data: dict[str, pd.DataFrame],
        test_data: dict[str, pd.DataFrame],
    ) -> FoldResult:
        """
        Execute a single walk-forward fold: fit HMM on train, simulate on test.

        Parameters
        ----------
        fold_index : int
        train_data : dict[str, pd.DataFrame]
        test_data : dict[str, pd.DataFrame]

        Returns
        -------
        FoldResult
        """
        raise NotImplementedError

    def _simulate_bar(
        self,
        date: pd.Timestamp,
        signals: dict[str, "TradingSignal"],  # noqa: F821
        prices: dict[str, float],
        portfolio: dict,
    ) -> list[dict]:
        """
        Simulate order execution for a single bar.

        Parameters
        ----------
        date : pd.Timestamp
        signals : dict[str, TradingSignal]
        prices : dict[str, float]
            Close prices used for execution (+ slippage applied internally).
        portfolio : dict
            Mutable portfolio state dict: {equity, cash, positions}.

        Returns
        -------
        list[dict]
            Executed trade records for this bar.
        """
        raise NotImplementedError

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to an execution price.

        Parameters
        ----------
        price : float
        side : str
            'buy' → price increases, 'sell' → price decreases.

        Returns
        -------
        float
        """
        raise NotImplementedError
