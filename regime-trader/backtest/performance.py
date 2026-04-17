"""
performance.py — Backtest performance metrics and reporting.

Responsibilities:
  - Compute Sharpe, Sortino, Calmar ratios from an equity curve.
  - Compute max drawdown, drawdown duration, and recovery time.
  - Break down returns and trade counts by regime label.
  - Compare strategy against buy-and-hold and trend benchmarks.
  - Export CSV reports and print Rich terminal tables.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoreMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    max_dd_duration_days: int
    longest_underwater_days: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_holding_days: float
    n_trades: int
    worst_day_pct: float
    worst_week_pct: float
    worst_month_pct: float
    max_consecutive_losses: int


@dataclass
class RegimeRow:
    regime: str
    pct_time: float
    total_return_pct: float
    avg_daily_return_pct: float
    sharpe: float
    n_trades: int
    win_rate_pct: float


@dataclass
class ConfidenceBucket:
    bucket: str          # e.g. "<50%", "50-60%", "60-70%", "70%+"
    pct_time: float
    avg_daily_return_pct: float
    sharpe: float


@dataclass
class BenchmarkRow:
    name: str
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float


@dataclass
class PerformanceReport:
    core: CoreMetrics
    regime_breakdown: list[RegimeRow]
    confidence_buckets: list[ConfidenceBucket]
    benchmarks: list[BenchmarkRow]


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────

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
        trading_days_per_year: int = _TRADING_DAYS,
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
        random_seeds: int = 100,
    ) -> PerformanceReport:
        """
        Compute the full performance report for a backtest result.

        Parameters
        ----------
        result : BacktestResult
        benchmark_ohlcv : pd.DataFrame | None
            OHLCV for the benchmark (e.g. SPY).  If provided, buy-and-hold
            and 200-SMA trend benchmarks are computed from it.
        random_seeds : int
            Number of Monte Carlo random-allocation seeds for benchmark.

        Returns
        -------
        PerformanceReport
        """
        equity = result.equity_curve
        trades_df = self._trades_to_df(result.trades)
        regime_series = result.regime_labels
        conf_series = result.confidence_history

        core = self._compute_core(equity, trades_df)
        regime_rows = self._regime_breakdown(equity, regime_series, trades_df)
        conf_buckets = self._confidence_buckets(equity, conf_series)
        benchmarks = self._compute_benchmarks(
            equity, benchmark_ohlcv, random_seeds
        )

        return PerformanceReport(
            core=core,
            regime_breakdown=regime_rows,
            confidence_buckets=conf_buckets,
            benchmarks=benchmarks,
        )

    def sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """Annualised Sharpe ratio."""
        rets = self._daily_returns(equity_curve)
        if len(rets) < 2:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess = rets - daily_rf
        std = excess.std(ddof=1)
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(self.trading_days_per_year))

    def sortino_ratio(self, equity_curve: pd.Series) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        rets = self._daily_returns(equity_curve)
        if len(rets) < 2:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess = rets - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std == 0:
            return 0.0
        return float(excess.mean() / downside_std * np.sqrt(self.trading_days_per_year))

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """Calmar = CAGR / max_drawdown."""
        mdd = self.max_drawdown(equity_curve)
        if mdd == 0:
            return float("inf")
        return float(self._annualised_return(equity_curve) / mdd)

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Maximum peak-to-trough drawdown as a positive fraction."""
        if len(equity_curve) < 2:
            return 0.0
        rolling_max = equity_curve.cummax()
        dd = (equity_curve - rolling_max) / rolling_max
        return float(-dd.min())

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Longest drawdown duration in calendar days (peak-to-recovery)."""
        if len(equity_curve) < 2:
            return 0
        rolling_max = equity_curve.cummax()
        is_underwater = equity_curve < rolling_max

        max_dur = 0
        start: Optional[pd.Timestamp] = None
        for date, uw in is_underwater.items():
            if uw and start is None:
                start = date
            elif not uw and start is not None:
                dur = (date - start).days
                max_dur = max(max_dur, dur)
                start = None
        if start is not None:
            dur = (equity_curve.index[-1] - start).days
            max_dur = max(max_dur, dur)
        return max_dur

    def regime_breakdown(
        self,
        equity_curve: pd.Series,
        regime_labels: pd.Series,
        trades: pd.DataFrame,
    ) -> list[RegimeRow]:
        """Per-regime performance statistics (public alias)."""
        return self._regime_breakdown(equity_curve, regime_labels, trades)

    def print_report(self, report: PerformanceReport) -> None:
        """Print a formatted report to the terminal using Rich (fallback to plain text)."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            self._print_rich(console, report)
        except ImportError:
            self._print_plain(report)

    def export_csv(
        self,
        result: BacktestResult,
        report: PerformanceReport,
        output_dir: str = ".",
    ) -> dict[str, Path]:
        """
        Write four CSV files to output_dir.

        Returns
        -------
        dict[str, Path]
            Keys: equity_curve, trade_log, regime_history, benchmark_comparison
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        # equity_curve.csv
        p = out / "equity_curve.csv"
        result.equity_curve.to_frame("equity").to_csv(p)
        paths["equity_curve"] = p

        # trade_log.csv
        p = out / "trade_log.csv"
        trades_df = self._trades_to_df(result.trades)
        trades_df.to_csv(p, index=False)
        paths["trade_log"] = p

        # regime_history.csv
        p = out / "regime_history.csv"
        hist_df = pd.DataFrame({
            "regime": result.regime_labels,
            "confidence": result.confidence_history,
        })
        hist_df.to_csv(p)
        paths["regime_history"] = p

        # benchmark_comparison.csv
        p = out / "benchmark_comparison.csv"
        rows = [
            {
                "name": b.name,
                "total_return_pct": round(b.total_return_pct, 4),
                "cagr_pct": round(b.cagr_pct, 4),
                "sharpe": round(b.sharpe, 4),
                "max_drawdown_pct": round(b.max_drawdown_pct, 4),
            }
            for b in report.benchmarks
        ]
        # prepend strategy row
        c = report.core
        rows.insert(0, {
            "name": "Strategy",
            "total_return_pct": round(c.total_return_pct, 4),
            "cagr_pct": round(c.cagr_pct, 4),
            "sharpe": round(c.sharpe, 4),
            "max_drawdown_pct": round(c.max_drawdown_pct, 4),
        })
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths["benchmark_comparison"] = p

        logger.info("CSVs written to %s", out)
        return paths

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _compute_core(
        self, equity: pd.Series, trades_df: pd.DataFrame
    ) -> CoreMetrics:
        rets = self._daily_returns(equity)

        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        cagr = self._annualised_return(equity) * 100
        mdd = self.max_drawdown(equity) * 100
        dd_dur = self.max_drawdown_duration(equity)
        longest_uw = self._longest_underwater(equity)

        sharpe = self.sharpe_ratio(equity)
        sortino = self.sortino_ratio(equity)
        calmar = self.calmar_ratio(equity)

        # Trade-level stats
        if len(trades_df) > 0 and "pnl_pct" in trades_df.columns:
            pnl = trades_df["pnl_pct"].dropna()
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            win_rate = len(wins) / len(pnl) * 100 if len(pnl) else 0.0
            avg_win = float(wins.mean() * 100) if len(wins) else 0.0
            avg_loss = float(losses.mean() * 100) if len(losses) else 0.0
            gross_profit = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            n_trades = len(pnl)
        else:
            win_rate = avg_win = avg_loss = 0.0
            profit_factor = 0.0
            n_trades = len(trades_df)

        avg_holding = self._avg_holding_days(trades_df)
        max_consec_loss = self._max_consecutive_losses(trades_df)

        # Worst periods
        worst_day = float(rets.min() * 100) if len(rets) else 0.0
        weekly = equity.resample("W").last().pct_change().dropna()
        worst_week = float(weekly.min() * 100) if len(weekly) else 0.0
        monthly = equity.resample("ME").last().pct_change().dropna()
        worst_month = float(monthly.min() * 100) if len(monthly) else 0.0

        return CoreMetrics(
            total_return_pct=round(total_ret, 4),
            cagr_pct=round(cagr, 4),
            sharpe=round(sharpe, 4),
            sortino=round(sortino, 4),
            calmar=round(calmar, 4),
            max_drawdown_pct=round(mdd, 4),
            max_dd_duration_days=dd_dur,
            longest_underwater_days=longest_uw,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 4),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            avg_holding_days=round(avg_holding, 2),
            n_trades=n_trades,
            worst_day_pct=round(worst_day, 4),
            worst_week_pct=round(worst_week, 4),
            worst_month_pct=round(worst_month, 4),
            max_consecutive_losses=max_consec_loss,
        )

    def _regime_breakdown(
        self,
        equity: pd.Series,
        regime_labels: Optional[pd.Series],
        trades_df: pd.DataFrame,
    ) -> list[RegimeRow]:
        if regime_labels is None or len(regime_labels) == 0:
            return []

        rets = self._daily_returns(equity)
        total_bars = len(regime_labels)
        rows: list[RegimeRow] = []

        for regime in regime_labels.unique():
            mask = regime_labels == regime
            pct_time = mask.sum() / total_bars * 100

            common_idx = rets.index.intersection(regime_labels.index[mask])
            regime_rets = rets.loc[common_idx]
            total_ret = float(((1 + regime_rets).prod() - 1) * 100)
            avg_daily = float(regime_rets.mean() * 100) if len(regime_rets) else 0.0

            daily_rf = self.risk_free_rate / self.trading_days_per_year
            excess = regime_rets - daily_rf
            if len(excess) > 1 and excess.std(ddof=1) > 0:
                regime_sharpe = float(
                    excess.mean() / excess.std(ddof=1) * np.sqrt(self.trading_days_per_year)
                )
            else:
                regime_sharpe = 0.0

            # Trade stats for this regime
            if len(trades_df) > 0 and "regime" in trades_df.columns:
                r_trades = trades_df[trades_df["regime"] == regime]
                n_trades = len(r_trades)
                if n_trades > 0 and "pnl_pct" in r_trades.columns:
                    pnl = r_trades["pnl_pct"].dropna()
                    win_rate = float((pnl > 0).mean() * 100) if len(pnl) else 0.0
                else:
                    win_rate = 0.0
            else:
                n_trades = 0
                win_rate = 0.0

            rows.append(RegimeRow(
                regime=str(regime),
                pct_time=round(pct_time, 2),
                total_return_pct=round(total_ret, 4),
                avg_daily_return_pct=round(avg_daily, 4),
                sharpe=round(regime_sharpe, 4),
                n_trades=n_trades,
                win_rate_pct=round(win_rate, 2),
            ))

        rows.sort(key=lambda r: r.total_return_pct, reverse=True)
        return rows

    def _confidence_buckets(
        self,
        equity: pd.Series,
        conf_series: Optional[pd.Series],
    ) -> list[ConfidenceBucket]:
        if conf_series is None or len(conf_series) == 0:
            return []

        rets = self._daily_returns(equity)
        buckets = [
            ("<50%",  lambda c: c < 0.50),
            ("50-60%", lambda c: 0.50 <= c < 0.60),
            ("60-70%", lambda c: 0.60 <= c < 0.70),
            ("70%+",  lambda c: c >= 0.70),
        ]
        results: list[ConfidenceBucket] = []
        total = len(conf_series)

        for label, pred in buckets:
            mask = conf_series.apply(pred)
            pct_time = mask.sum() / total * 100
            common = rets.index.intersection(conf_series.index[mask])
            bucket_rets = rets.loc[common]
            avg_daily = float(bucket_rets.mean() * 100) if len(bucket_rets) else 0.0
            daily_rf = self.risk_free_rate / self.trading_days_per_year
            excess = bucket_rets - daily_rf
            if len(excess) > 1 and excess.std(ddof=1) > 0:
                sharpe = float(
                    excess.mean() / excess.std(ddof=1) * np.sqrt(self.trading_days_per_year)
                )
            else:
                sharpe = 0.0
            results.append(ConfidenceBucket(
                bucket=label,
                pct_time=round(pct_time, 2),
                avg_daily_return_pct=round(avg_daily, 4),
                sharpe=round(sharpe, 4),
            ))

        return results

    def _compute_benchmarks(
        self,
        equity: pd.Series,
        benchmark_ohlcv: Optional[pd.DataFrame],
        random_seeds: int,
    ) -> list[BenchmarkRow]:
        rows: list[BenchmarkRow] = []

        if benchmark_ohlcv is not None:
            bm_close = benchmark_ohlcv["close"].reindex(equity.index).ffill()
            bah_equity = equity.iloc[0] * bm_close / bm_close.iloc[0]
            rows.append(self._equity_to_benchmark_row("Buy & Hold", bah_equity))

            # 200-SMA trend: invested when close > SMA200, else cash
            sma200 = bm_close.rolling(200, min_periods=1).mean()
            invested = (bm_close > sma200).astype(float)
            daily_bm = bm_close.pct_change().fillna(0.0)
            trend_rets = daily_bm * invested.shift(1).fillna(0.0)
            trend_equity = equity.iloc[0] * (1 + trend_rets).cumprod()
            rows.append(self._equity_to_benchmark_row("200-SMA Trend", trend_equity))

        # Random allocation (Monte Carlo)
        if random_seeds > 0 and benchmark_ohlcv is not None:
            bm_close = benchmark_ohlcv["close"].reindex(equity.index).ffill()
            daily_bm = bm_close.pct_change().fillna(0.0)
            rng = np.random.default_rng(42)
            mc_sharpes: list[float] = []
            mc_returns: list[float] = []
            mc_mdds: list[float] = []
            mc_cagrs: list[float] = []
            for seed in range(random_seeds):
                alloc = rng.uniform(0.3, 1.0, len(daily_bm))
                rand_rets = daily_bm.values * alloc
                rand_eq = pd.Series(
                    equity.iloc[0] * np.cumprod(1 + rand_rets),
                    index=equity.index,
                )
                mc_sharpes.append(self.sharpe_ratio(rand_eq))
                mc_returns.append((rand_eq.iloc[-1] / rand_eq.iloc[0] - 1) * 100)
                mc_mdds.append(self.max_drawdown(rand_eq) * 100)
                mc_cagrs.append(self._annualised_return(rand_eq) * 100)
            rows.append(BenchmarkRow(
                name=f"Random Alloc (n={random_seeds}, median)",
                total_return_pct=round(float(np.median(mc_returns)), 4),
                cagr_pct=round(float(np.median(mc_cagrs)), 4),
                sharpe=round(float(np.median(mc_sharpes)), 4),
                max_drawdown_pct=round(float(np.median(mc_mdds)), 4),
            ))

        return rows

    def _equity_to_benchmark_row(self, name: str, eq: pd.Series) -> BenchmarkRow:
        return BenchmarkRow(
            name=name,
            total_return_pct=round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 4),
            cagr_pct=round(self._annualised_return(eq) * 100, 4),
            sharpe=round(self.sharpe_ratio(eq), 4),
            max_drawdown_pct=round(self.max_drawdown(eq) * 100, 4),
        )

    def _daily_returns(self, equity_curve: pd.Series) -> pd.Series:
        return equity_curve.pct_change().dropna()

    def _annualised_return(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        total = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / self.trading_days_per_year
        if n_years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    def _longest_underwater(self, equity_curve: pd.Series) -> int:
        """Longest continuous period below a prior peak (calendar days)."""
        rolling_max = equity_curve.cummax()
        is_uw = equity_curve < rolling_max
        max_dur = 0
        cur_dur = 0
        prev_date: Optional[pd.Timestamp] = None
        for date, uw in is_uw.items():
            if uw:
                if prev_date is not None:
                    cur_dur += (date - prev_date).days
                else:
                    cur_dur = 0
                prev_date = date
            else:
                max_dur = max(max_dur, cur_dur)
                cur_dur = 0
                prev_date = None
        max_dur = max(max_dur, cur_dur)
        return max_dur

    def _avg_holding_days(self, trades_df: pd.DataFrame) -> float:
        if len(trades_df) == 0:
            return 0.0
        if "signal_date" in trades_df.columns and "exec_date" in trades_df.columns:
            holding = (
                pd.to_datetime(trades_df["exec_date"])
                - pd.to_datetime(trades_df["signal_date"])
            ).dt.days
            return float(holding.mean())
        return 0.0

    def _max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        if len(trades_df) == 0 or "pnl_pct" not in trades_df.columns:
            return 0
        pnl = trades_df["pnl_pct"].dropna()
        max_consec = 0
        cur = 0
        for v in pnl:
            if v < 0:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        return max_consec

    def _trades_to_df(self, trades: list) -> pd.DataFrame:
        """Convert list of TradeRecord to DataFrame, add pnl_pct if possible."""
        if not trades:
            return pd.DataFrame()
        records = [t.__dict__ if hasattr(t, "__dict__") else dict(t) for t in trades]
        df = pd.DataFrame(records)

        # Compute PnL% by pairing buys and sells on same symbol
        if "side" in df.columns and "exec_price" in df.columns and "symbol" in df.columns:
            df["pnl_pct"] = float("nan")
            for sym in df["symbol"].unique():
                sym_df = df[df["symbol"] == sym].copy()
                buys = sym_df[sym_df["side"] == "BUY"]["exec_price"].values
                sells = sym_df[sym_df["side"] == "SELL"]["exec_price"].values
                min_pairs = min(len(buys), len(sells))
                if min_pairs > 0:
                    pair_pnl = (sells[:min_pairs] - buys[:min_pairs]) / buys[:min_pairs]
                    buy_idx = sym_df[sym_df["side"] == "BUY"].index[:min_pairs]
                    df.loc[buy_idx, "pnl_pct"] = pair_pnl
        return df

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def _print_rich(self, console, report: PerformanceReport) -> None:
        from rich.table import Table

        c = report.core

        # Core metrics table
        t = Table(title="[bold cyan]Strategy Performance[/bold cyan]", show_header=True)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        rows = [
            ("Total Return", f"{c.total_return_pct:.2f}%"),
            ("CAGR", f"{c.cagr_pct:.2f}%"),
            ("Sharpe Ratio", f"{c.sharpe:.3f}"),
            ("Sortino Ratio", f"{c.sortino:.3f}"),
            ("Calmar Ratio", f"{c.calmar:.3f}"),
            ("Max Drawdown", f"{c.max_drawdown_pct:.2f}%"),
            ("Max DD Duration", f"{c.max_dd_duration_days} days"),
            ("Longest Underwater", f"{c.longest_underwater_days} days"),
            ("Win Rate", f"{c.win_rate_pct:.1f}%"),
            ("Profit Factor", f"{c.profit_factor:.3f}"),
            ("Avg Win", f"{c.avg_win_pct:.2f}%"),
            ("Avg Loss", f"{c.avg_loss_pct:.2f}%"),
            ("Avg Holding", f"{c.avg_holding_days:.1f} days"),
            ("N Trades", str(c.n_trades)),
            ("Worst Day", f"{c.worst_day_pct:.2f}%"),
            ("Worst Week", f"{c.worst_week_pct:.2f}%"),
            ("Worst Month", f"{c.worst_month_pct:.2f}%"),
            ("Max Consec. Losses", str(c.max_consecutive_losses)),
        ]
        for metric, value in rows:
            t.add_row(metric, value)
        console.print(t)

        # Regime breakdown
        if report.regime_breakdown:
            rt = Table(title="[bold cyan]Regime Breakdown[/bold cyan]", show_header=True)
            for col in ["Regime", "% Time", "Total Ret%", "Avg Daily%", "Sharpe", "Trades", "Win%"]:
                rt.add_column(col, justify="right" if col != "Regime" else "left")
            for r in report.regime_breakdown:
                rt.add_row(
                    r.regime,
                    f"{r.pct_time:.1f}",
                    f"{r.total_return_pct:.2f}",
                    f"{r.avg_daily_return_pct:.4f}",
                    f"{r.sharpe:.3f}",
                    str(r.n_trades),
                    f"{r.win_rate_pct:.1f}",
                )
            console.print(rt)

        # Confidence buckets
        if report.confidence_buckets:
            ct = Table(title="[bold cyan]Confidence Buckets[/bold cyan]", show_header=True)
            for col in ["Bucket", "% Time", "Avg Daily%", "Sharpe"]:
                ct.add_column(col, justify="right" if col != "Bucket" else "left")
            for b in report.confidence_buckets:
                ct.add_row(
                    b.bucket,
                    f"{b.pct_time:.1f}",
                    f"{b.avg_daily_return_pct:.4f}",
                    f"{b.sharpe:.3f}",
                )
            console.print(ct)

        # Benchmarks
        if report.benchmarks:
            bt = Table(title="[bold cyan]Benchmark Comparison[/bold cyan]", show_header=True)
            for col in ["Benchmark", "Total Ret%", "CAGR%", "Sharpe", "Max DD%"]:
                bt.add_column(col, justify="right" if col != "Benchmark" else "left")
            for b in report.benchmarks:
                bt.add_row(
                    b.name,
                    f"{b.total_return_pct:.2f}",
                    f"{b.cagr_pct:.2f}",
                    f"{b.sharpe:.3f}",
                    f"{b.max_drawdown_pct:.2f}",
                )
            console.print(bt)

    def _print_plain(self, report: PerformanceReport) -> None:
        c = report.core
        lines = [
            "=== Strategy Performance ===",
            f"  Total Return     : {c.total_return_pct:.2f}%",
            f"  CAGR             : {c.cagr_pct:.2f}%",
            f"  Sharpe           : {c.sharpe:.3f}",
            f"  Sortino          : {c.sortino:.3f}",
            f"  Calmar           : {c.calmar:.3f}",
            f"  Max Drawdown     : {c.max_drawdown_pct:.2f}%",
            f"  Max DD Duration  : {c.max_dd_duration_days} days",
            f"  Win Rate         : {c.win_rate_pct:.1f}%",
            f"  Profit Factor    : {c.profit_factor:.3f}",
            f"  N Trades         : {c.n_trades}",
            f"  Worst Day        : {c.worst_day_pct:.2f}%",
        ]
        print("\n".join(lines))

        if report.regime_breakdown:
            print("\n=== Regime Breakdown ===")
            header = f"  {'Regime':<20} {'%Time':>6} {'TotRet%':>8} {'Sharpe':>7}"
            print(header)
            for r in report.regime_breakdown:
                print(f"  {r.regime:<20} {r.pct_time:>6.1f} {r.total_return_pct:>8.2f} {r.sharpe:>7.3f}")

        if report.benchmarks:
            print("\n=== Benchmarks ===")
            for b in report.benchmarks:
                print(f"  {b.name:<30} Ret={b.total_return_pct:.2f}% Sharpe={b.sharpe:.3f} MDD={b.max_drawdown_pct:.2f}%")
