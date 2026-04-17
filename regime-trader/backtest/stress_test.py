"""
stress_test.py — Crash injection and gap simulation for regime robustness testing.

Responsibilities:
  - Inject synthetic market crashes into historical OHLCV data.
  - Simulate overnight price gaps.
  - Test the trading system's behaviour under extreme volatility regimes.
  - Return stress-test equity curves and regime responses for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult, WalkForwardBacktester


@dataclass
class CrashScenario:
    """Parameters for a single synthetic crash injection."""

    name: str
    start_date: pd.Timestamp
    duration_days: int
    peak_drawdown: float          # e.g. 0.40 for a 40% crash
    recovery_days: Optional[int]  # None = no recovery within the dataset
    volatility_multiplier: float  # Additional vol amplification during crash


@dataclass
class GapScenario:
    """Parameters for an overnight gap simulation."""

    name: str
    date: pd.Timestamp
    gap_pct: float               # Positive = gap up, negative = gap down
    symbols: Optional[list[str]] = None   # None = apply to all symbols


class StressTester:
    """
    Injects synthetic stress scenarios into historical data and re-runs backtest.

    Parameters
    ----------
    backtester : WalkForwardBacktester
        Configured backtester instance to re-run under stressed data.
    """

    def __init__(self, backtester: WalkForwardBacktester) -> None:
        self.backtester = backtester

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_crash(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        scenario: CrashScenario,
    ) -> BacktestResult:
        """
        Inject a synthetic crash into `ohlcv_by_symbol` and run the backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
            Original historical data.
        scenario : CrashScenario

        Returns
        -------
        BacktestResult
        """
        raise NotImplementedError

    def run_gap(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        scenario: GapScenario,
    ) -> BacktestResult:
        """
        Inject an overnight gap into `ohlcv_by_symbol` and run the backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        scenario : GapScenario

        Returns
        -------
        BacktestResult
        """
        raise NotImplementedError

    def run_all_scenarios(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        crash_scenarios: list[CrashScenario],
        gap_scenarios: list[GapScenario],
    ) -> dict[str, BacktestResult]:
        """
        Run the backtest under each scenario and return all results keyed by name.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        crash_scenarios : list[CrashScenario]
        gap_scenarios : list[GapScenario]

        Returns
        -------
        dict[str, BacktestResult]
            scenario.name → BacktestResult
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Data modification helpers
    # ------------------------------------------------------------------

    def _inject_crash(
        self,
        ohlcv: pd.DataFrame,
        scenario: CrashScenario,
    ) -> pd.DataFrame:
        """
        Modify `ohlcv` to simulate a crash according to `scenario`.

        Scales returns to produce the target peak_drawdown over duration_days,
        applies volatility_multiplier to bar high-low ranges, and optionally
        adds a recovery path.

        Parameters
        ----------
        ohlcv : pd.DataFrame
        scenario : CrashScenario

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError

    def _inject_gap(
        self,
        ohlcv: pd.DataFrame,
        scenario: GapScenario,
    ) -> pd.DataFrame:
        """
        Apply a single overnight gap to `ohlcv` at scenario.date.

        Multiplies open price on the gap date by (1 + gap_pct), then adjusts
        all subsequent bars to maintain relative returns.

        Parameters
        ----------
        ohlcv : pd.DataFrame
        scenario : GapScenario

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError
