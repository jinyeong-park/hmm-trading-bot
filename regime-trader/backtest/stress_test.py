"""
stress_test.py — Crash injection and gap simulation for regime robustness testing.

Responsibilities:
  - Inject synthetic market crashes into historical OHLCV data.
  - Simulate overnight price gaps.
  - Monte Carlo crash injection at random points (100 seeds).
  - Test regime misclassification robustness by shuffling labels.
  - Return stress-test equity curves and summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult, WalkForwardBacktester

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclasses
# ─────────────────────────────────────────────────────────────────────────────

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


@dataclass
class MonteCarloCrashResult:
    """Aggregated stats from 100 Monte Carlo crash simulations."""
    n_simulations: int
    crash_pct_min: float         # min single-day crash magnitude injected
    crash_pct_max: float         # max single-day crash magnitude injected
    final_equity_mean: float
    final_equity_median: float
    final_equity_p5: float       # 5th-percentile (tail)
    max_drawdown_mean: float
    max_drawdown_p95: float      # worst-case 95th-percentile drawdown
    pct_survived: float          # fraction that didn't blow up (equity > 0)
    equity_curves: list[pd.Series] = field(default_factory=list)


@dataclass
class StressResult:
    """Single stress scenario outcome."""
    scenario_name: str
    backtest_result: BacktestResult
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main stress tester
# ─────────────────────────────────────────────────────────────────────────────

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
    ) -> StressResult:
        """
        Inject a synthetic crash into `ohlcv_by_symbol` and run the backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        scenario : CrashScenario

        Returns
        -------
        StressResult
        """
        stressed = {
            sym: self._inject_crash(ohlcv.copy(), scenario)
            for sym, ohlcv in ohlcv_by_symbol.items()
        }
        result = self.backtester.run(stressed)
        return self._wrap_result(scenario.name, result)

    def run_gap(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        scenario: GapScenario,
    ) -> StressResult:
        """
        Inject an overnight gap into `ohlcv_by_symbol` and run the backtest.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        scenario : GapScenario

        Returns
        -------
        StressResult
        """
        target_syms = set(scenario.symbols) if scenario.symbols else set(ohlcv_by_symbol.keys())
        stressed = {}
        for sym, ohlcv in ohlcv_by_symbol.items():
            if sym in target_syms:
                stressed[sym] = self._inject_gap(ohlcv.copy(), scenario)
            else:
                stressed[sym] = ohlcv.copy()
        result = self.backtester.run(stressed)
        return self._wrap_result(scenario.name, result)

    def run_all_scenarios(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        crash_scenarios: list[CrashScenario],
        gap_scenarios: list[GapScenario],
    ) -> dict[str, StressResult]:
        """
        Run the backtest under each scenario and return all results keyed by name.

        Returns
        -------
        dict[str, StressResult]
        """
        results: dict[str, StressResult] = {}
        for s in crash_scenarios:
            logger.info("Running crash scenario: %s", s.name)
            results[s.name] = self.run_crash(ohlcv_by_symbol, s)
        for s in gap_scenarios:
            logger.info("Running gap scenario: %s", s.name)
            results[s.name] = self.run_gap(ohlcv_by_symbol, s)
        return results

    def run_monte_carlo_crashes(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        n_simulations: int = 100,
        n_crash_points: int = 10,
        crash_pct_range: tuple[float, float] = (0.05, 0.15),
        duration_range: tuple[int, int] = (1, 5),
        seed: int = 42,
    ) -> MonteCarloCrashResult:
        """
        Insert -5% to -15% single-day gaps at `n_crash_points` random locations,
        repeated for `n_simulations` independent random seeds.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        n_simulations : int
            Number of independent Monte Carlo runs.
        n_crash_points : int
            Number of crash events to inject per simulation.
        crash_pct_range : tuple[float, float]
            (min, max) drawdown magnitude as positive fractions.
        duration_range : tuple[int, int]
            (min, max) crash duration in trading days.
        seed : int
            Base RNG seed.

        Returns
        -------
        MonteCarloCrashResult
        """
        rng = np.random.default_rng(seed)
        primary = list(ohlcv_by_symbol.keys())[0]
        available_dates = list(ohlcv_by_symbol[primary].index)

        final_equities: list[float] = []
        max_drawdowns: list[float] = []
        equity_curves: list[pd.Series] = []
        survived_count = 0

        for sim_idx in range(n_simulations):
            # Sample random crash dates and magnitudes
            date_indices = rng.choice(
                len(available_dates) - max(duration_range) - 1,
                size=n_crash_points,
                replace=False,
            )
            date_indices.sort()

            stressed = {sym: ohlcv.copy() for sym, ohlcv in ohlcv_by_symbol.items()}

            for di in date_indices:
                crash_date = available_dates[di]
                pct = float(rng.uniform(crash_pct_range[0], crash_pct_range[1]))
                dur = int(rng.integers(duration_range[0], duration_range[1] + 1))
                scenario = CrashScenario(
                    name=f"mc_{sim_idx}_{di}",
                    start_date=crash_date,
                    duration_days=dur,
                    peak_drawdown=pct,
                    recovery_days=None,
                    volatility_multiplier=2.0,
                )
                for sym in stressed:
                    stressed[sym] = self._inject_crash(stressed[sym], scenario)

            try:
                result = self.backtester.run(stressed)
                eq = result.equity_curve
                if len(eq) == 0:
                    continue
                final_eq = float(eq.iloc[-1])
                initial_eq = float(eq.iloc[0])
                mdd = self._compute_max_drawdown(eq)

                final_equities.append(final_eq)
                max_drawdowns.append(mdd * 100)
                equity_curves.append(eq)
                if final_eq > 0:
                    survived_count += 1

                logger.debug(
                    "MC sim %d/%d  final_equity=%.0f  mdd=%.1f%%",
                    sim_idx + 1, n_simulations, final_eq, mdd * 100,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MC sim %d failed: %s", sim_idx, exc)

        if not final_equities:
            return MonteCarloCrashResult(
                n_simulations=n_simulations,
                crash_pct_min=crash_pct_range[0] * 100,
                crash_pct_max=crash_pct_range[1] * 100,
                final_equity_mean=0.0,
                final_equity_median=0.0,
                final_equity_p5=0.0,
                max_drawdown_mean=0.0,
                max_drawdown_p95=0.0,
                pct_survived=0.0,
                equity_curves=[],
            )

        arr = np.array(final_equities)
        return MonteCarloCrashResult(
            n_simulations=n_simulations,
            crash_pct_min=crash_pct_range[0] * 100,
            crash_pct_max=crash_pct_range[1] * 100,
            final_equity_mean=round(float(arr.mean()), 2),
            final_equity_median=round(float(np.median(arr)), 2),
            final_equity_p5=round(float(np.percentile(arr, 5)), 2),
            max_drawdown_mean=round(float(np.mean(max_drawdowns)), 4),
            max_drawdown_p95=round(float(np.percentile(max_drawdowns, 95)), 4),
            pct_survived=round(survived_count / n_simulations * 100, 2),
            equity_curves=equity_curves,
        )

    def run_gap_risk(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        n_gaps: int = 20,
        atr_multiplier_range: tuple[float, float] = (2.0, 5.0),
        seed: int = 99,
    ) -> list[StressResult]:
        """
        Inject overnight gaps of 2–5× ATR at random points and collect results.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        n_gaps : int
            Number of gap events to test.
        atr_multiplier_range : tuple[float, float]
            (min, max) gap size as a multiple of 14-day ATR.
        seed : int

        Returns
        -------
        list[StressResult]
            One result per gap scenario.
        """
        rng = np.random.default_rng(seed)
        primary = list(ohlcv_by_symbol.keys())[0]
        primary_ohlcv = ohlcv_by_symbol[primary]
        available_dates = list(primary_ohlcv.index[20:])  # skip warmup

        atr = self._compute_atr(primary_ohlcv)
        results: list[StressResult] = []

        date_indices = rng.choice(len(available_dates), size=n_gaps, replace=False)

        for i, di in enumerate(date_indices):
            gap_date = available_dates[int(di)]
            mult = float(rng.uniform(atr_multiplier_range[0], atr_multiplier_range[1]))
            gap_price = float(primary_ohlcv.loc[gap_date, "close"]) if gap_date in primary_ohlcv.index else 100.0
            gap_pct = -(atr * mult / gap_price)  # always gap down for stress test

            scenario = GapScenario(
                name=f"gap_{i:02d}_{gap_date.date()}_{mult:.1f}x_ATR",
                date=gap_date,
                gap_pct=gap_pct,
            )
            logger.info("Gap scenario %d/%d: %s  gap=%.2f%%", i + 1, n_gaps, scenario.name, gap_pct * 100)
            results.append(self.run_gap(ohlcv_by_symbol, scenario))

        return results

    def run_regime_misclassification(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        shuffle_fraction: float = 0.20,
        n_trials: int = 10,
        seed: int = 7,
    ) -> list[StressResult]:
        """
        Simulate regime misclassification by randomly perturbing the OHLCV
        such that the HMM is more likely to misclassify states, then re-run
        the backtest to verify risk management contains the damage.

        The perturbation approach: randomly flip the sign of `shuffle_fraction`
        of daily returns (i.e. convert gains to losses and vice versa) to
        confuse the HMM's volatility clustering, without changing the overall
        distribution too much.

        Parameters
        ----------
        ohlcv_by_symbol : dict[str, pd.DataFrame]
        shuffle_fraction : float
            Fraction of bars whose returns are sign-flipped.
        n_trials : int
        seed : int

        Returns
        -------
        list[StressResult]
        """
        rng = np.random.default_rng(seed)
        results: list[StressResult] = []

        for trial in range(n_trials):
            stressed: dict[str, pd.DataFrame] = {}
            for sym, ohlcv in ohlcv_by_symbol.items():
                stressed[sym] = self._perturb_returns(ohlcv.copy(), shuffle_fraction, rng)
            try:
                result = self.backtester.run(stressed)
                sr = self._wrap_result(f"misclassify_trial_{trial}", result)
                sr.notes = f"shuffle_fraction={shuffle_fraction:.0%}"
                results.append(sr)
                logger.info(
                    "Misclassification trial %d/%d  ret=%.2f%%  mdd=%.2f%%",
                    trial + 1, n_trials, sr.total_return_pct, sr.max_drawdown_pct,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Misclassification trial %d failed: %s", trial, exc)

        return results

    # ------------------------------------------------------------------
    # Data modification helpers
    # ------------------------------------------------------------------

    def _inject_crash(
        self,
        ohlcv: pd.DataFrame,
        scenario: CrashScenario,
    ) -> pd.DataFrame:
        """
        Modify `ohlcv` to simulate a crash.

        The crash is modelled as a sequence of negative daily returns that
        compound to produce `peak_drawdown` over `duration_days`.  After
        the crash, an optional V-shaped recovery is appended.  During the
        crash window, high-low ranges are expanded by `volatility_multiplier`.

        Parameters
        ----------
        ohlcv : pd.DataFrame   (columns: open, high, low, close, volume)
        scenario : CrashScenario

        Returns
        -------
        pd.DataFrame
        """
        df = ohlcv.copy()
        if scenario.start_date not in df.index:
            # Find nearest available date
            valid = df.index[df.index >= scenario.start_date]
            if len(valid) == 0:
                return df
            start = valid[0]
        else:
            start = scenario.start_date

        start_pos = df.index.get_loc(start)
        dur = scenario.duration_days
        end_pos = min(start_pos + dur, len(df) - 1)
        crash_idx = df.index[start_pos: end_pos + 1]

        # Per-bar return to reach peak_drawdown over dur bars
        per_bar_ret = (1 - scenario.peak_drawdown) ** (1 / dur) - 1  # e.g. ~-0.087 for 40% in 5 days

        # Scale each bar in crash window
        cum_factor = 1.0
        for date in crash_idx:
            cum_factor *= (1 + per_bar_ret)
            prev_close = df.loc[date, "close"]
            new_close = prev_close * (1 + per_bar_ret)
            spread = (df.loc[date, "high"] - df.loc[date, "low"]) * scenario.volatility_multiplier
            df.loc[date, "close"] = new_close
            df.loc[date, "open"] = new_close * (1 + 0.002)  # small gap open
            df.loc[date, "high"] = new_close + spread * 0.3
            df.loc[date, "low"] = new_close - spread * 0.7

        # Adjust all bars after the crash window to maintain continuity
        if end_pos + 1 < len(df):
            # The price level shift at end of crash needs to propagate forward
            # as a level-shift (not as additional returns), so the strategy sees
            # the new lower price level going forward.
            price_at_crash_end = df.iloc[end_pos]["close"]
            orig_price_at_crash_end = ohlcv.iloc[end_pos]["close"]
            level_shift = price_at_crash_end / orig_price_at_crash_end if orig_price_at_crash_end > 0 else 1.0

            after_idx = df.index[end_pos + 1:]

            # Optional recovery: interpolate back to original price over recovery_days
            if scenario.recovery_days is not None and scenario.recovery_days > 0:
                rec_end = min(end_pos + 1 + scenario.recovery_days, len(df) - 1)
                rec_idx = df.index[end_pos + 1: rec_end + 1]
                for i, date in enumerate(rec_idx):
                    recovery_frac = (i + 1) / len(rec_idx)
                    shift = level_shift + (1.0 - level_shift) * recovery_frac
                    for col in ["open", "high", "low", "close"]:
                        df.loc[date, col] = ohlcv.loc[date, col] * shift
                remaining_after = df.index[rec_end + 1:]
            else:
                remaining_after = after_idx

            # Bars beyond the recovery window retain the level-shifted price
            for date in remaining_after:
                for col in ["open", "high", "low", "close"]:
                    df.loc[date, col] = ohlcv.loc[date, col] * level_shift

        return df

    def _inject_gap(
        self,
        ohlcv: pd.DataFrame,
        scenario: GapScenario,
    ) -> pd.DataFrame:
        """
        Apply a single overnight gap to `ohlcv` at `scenario.date`.

        The open of the gap bar is shifted by `gap_pct`.  All subsequent bars
        are level-shifted by the same factor so prior-day close ≠ next-day open
        (the gap is permanent; we're testing position-sizing under gap risk, not
        mean reversion).

        Parameters
        ----------
        ohlcv : pd.DataFrame
        scenario : GapScenario

        Returns
        -------
        pd.DataFrame
        """
        df = ohlcv.copy()
        if scenario.date not in df.index:
            valid = df.index[df.index >= scenario.date]
            if len(valid) == 0:
                return df
            gap_date = valid[0]
        else:
            gap_date = scenario.date

        gap_pos = df.index.get_loc(gap_date)
        factor = 1.0 + scenario.gap_pct

        # Apply gap: shift open/high/low/close for the gap bar and all after it
        gap_forward = df.index[gap_pos:]
        for col in ["open", "high", "low", "close"]:
            df.loc[gap_forward, col] = df.loc[gap_forward, col] * factor

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_result(self, name: str, result: BacktestResult) -> StressResult:
        eq = result.equity_curve
        if len(eq) < 2:
            return StressResult(
                scenario_name=name,
                backtest_result=result,
                final_equity=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
            )
        final = float(eq.iloc[-1])
        initial = float(eq.iloc[0])
        total_ret = (final / initial - 1) * 100 if initial > 0 else 0.0
        mdd = self._compute_max_drawdown(eq) * 100
        return StressResult(
            scenario_name=name,
            backtest_result=result,
            final_equity=round(final, 2),
            total_return_pct=round(total_ret, 4),
            max_drawdown_pct=round(mdd, 4),
        )

    def _compute_max_drawdown(self, equity: pd.Series) -> float:
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / rolling_max
        return float(-dd.min())

    def _compute_atr(self, ohlcv: pd.DataFrame, window: int = 14) -> float:
        """14-day ATR as a price-level value (not percentage)."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        prev_close = ohlcv["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=window, adjust=False).mean()
        return float(atr.iloc[-1]) if len(atr) else 1.0

    def _perturb_returns(
        self,
        ohlcv: pd.DataFrame,
        fraction: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Sign-flip `fraction` of daily close-to-close returns to simulate
        a scenario where the HMM receives misleading volatility signals.
        """
        df = ohlcv.copy()
        n = len(df)
        n_flip = max(1, int(n * fraction))
        flip_indices = rng.choice(n - 1, size=n_flip, replace=False) + 1  # skip first bar

        for pos in flip_indices:
            prev_close = df.iloc[pos - 1]["close"]
            orig_close = df.iloc[pos]["close"]
            daily_ret = (orig_close / prev_close) - 1 if prev_close > 0 else 0.0
            new_close = prev_close * (1 - daily_ret)  # sign-flip return
            if new_close <= 0:
                continue
            date = df.index[pos]
            scale = new_close / orig_close if orig_close > 0 else 1.0
            df.loc[date, "close"] = new_close
            df.loc[date, "high"] = max(df.loc[date, "open"], new_close) * (1 + 0.003)
            df.loc[date, "low"] = min(df.loc[date, "open"], new_close) * (1 - 0.003)

        return df
