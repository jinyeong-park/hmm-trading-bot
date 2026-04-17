"""
performance.py — Backtest performance metrics and reporting.

Responsibilities:
  - Compute Sharpe, Sortino, Calmar ratios from an equity curve.
  - Compute max drawdown, drawdown duration, and recovery time.
  - Break down returns and trade counts by regime label.
  - Compare strategy against a buy-and-hold benchmark.
  - Generate a summary dict / DataFrame report.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult


class PerformanceAnalyser:
    """
    Computes performance metrics from a BacktestResult.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate used for Sharpe / Sortino calculation.
    trading_days_per_year : int
        Annualisation factor (252 for US equities).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        trading_days_per_year: int = 252,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        result: BacktestResult,
        benchmark_ohlcv: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Compute the full performance report for a backtest result.

        Parameters
        ----------
        result : BacktestResult
        benchmark_ohlcv : pd.DataFrame | None
            OHLCV for the benchmark (e.g. SPY). If provided, benchmark metrics
            are included in the report.

        Returns
        -------
        dict
            Keys include: sharpe, sortino, calmar, max_drawdown,
            max_dd_duration_days, total_return, annualised_return,
            win_rate, profit_factor, regime_breakdown, benchmark_sharpe, etc.
        """
        raise NotImplementedError

    def sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """
        Compute annualised Sharpe ratio from an equity curve.

        Parameters
        ----------
        equity_curve : pd.Series
            DatetimeIndex → equity values.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def sortino_ratio(self, equity_curve: pd.Series) -> float:
        """
        Compute annualised Sortino ratio (downside deviation only).

        Parameters
        ----------
        equity_curve : pd.Series

        Returns
        -------
        float
        """
        raise NotImplementedError

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """
        Compute Calmar ratio = annualised return / max drawdown.

        Parameters
        ----------
        equity_curve : pd.Series

        Returns
        -------
        float
        """
        raise NotImplementedError

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Compute maximum peak-to-trough drawdown as a positive fraction.

        Parameters
        ----------
        equity_curve : pd.Series

        Returns
        -------
        float
        """
        raise NotImplementedError

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """
        Compute the longest drawdown duration in calendar days.

        Parameters
        ----------
        equity_curve : pd.Series

        Returns
        -------
        int
        """
        raise NotImplementedError

    def regime_breakdown(
        self,
        equity_curve: pd.Series,
        regime_labels: pd.Series,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-regime performance statistics.

        Parameters
        ----------
        equity_curve : pd.Series
        regime_labels : pd.Series
            DatetimeIndex → regime label string.
        trades : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Rows = regimes, columns = [n_trades, win_rate, avg_return, total_return].
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _daily_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Compute percentage daily returns from an equity curve."""
        raise NotImplementedError

    def _annualised_return(self, equity_curve: pd.Series) -> float:
        """Compute CAGR from an equity curve."""
        raise NotImplementedError
