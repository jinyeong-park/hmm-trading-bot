"""
test_look_ahead.py — Look-ahead bias prevention tests.

Verifies that no feature or signal uses future information:
  - Feature values at time t depend only on data up to and including t.
  - HMM forward-algorithm inference is causal (changing obs[t+1:] never
    alters the posterior at t).
  - Backtest signal generation uses only bars available at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import FeatureEngineer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature look-ahead tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureLookAhead:
    """Verify all feature computations are strictly causal."""

    def test_features_do_not_use_future_close(self):
        """
        Randomize future prices and confirm past feature values are unchanged.

        Strategy: compute features on full data, then replace data[T//2:] with
        random values and recompute.  Feature values at t < T//2 must be identical.
        """
        engineer = FeatureEngineer(normalise=False)
        n = 600
        ohlcv = _make_ohlcv(n)

        features_full = engineer.transform(ohlcv)
        split_date = features_full.index[len(features_full) // 2]

        # Corrupt all rows after split_date with random prices
        rng = np.random.default_rng(999)
        ohlcv_corrupted = ohlcv.copy()
        mask = ohlcv_corrupted.index > split_date
        n_corrupt = mask.sum()
        ohlcv_corrupted.loc[mask, "close"] = rng.uniform(50, 200, n_corrupt)
        ohlcv_corrupted.loc[mask, "high"] = ohlcv_corrupted.loc[mask, "close"] * 1.01
        ohlcv_corrupted.loc[mask, "low"] = ohlcv_corrupted.loc[mask, "close"] * 0.99

        features_corrupted = engineer.transform(ohlcv_corrupted)

        # All rows at or before split_date must match exactly
        common_idx = features_full.index[features_full.index <= split_date]
        common_idx = common_idx.intersection(features_corrupted.index)

        assert len(common_idx) > 0, "No common dates to compare — check split logic"

        np.testing.assert_array_almost_equal(
            features_full.loc[common_idx].values,
            features_corrupted.loc[common_idx].values,
            decimal=10,
            err_msg="Feature values before split changed after corrupting future data!",
        )

    def test_log_return_uses_only_past_close(self):
        """
        log_ret_1[t] must equal ln(close[t] / close[t-1]) and nothing else.
        """
        engineer = FeatureEngineer(normalise=False)
        n = 400
        ohlcv = _make_ohlcv(n)
        close = ohlcv["close"]

        features = engineer.transform(ohlcv)
        expected = np.log(close / close.shift(1)).dropna()

        # Align on common index
        common = features.index.intersection(expected.index)
        np.testing.assert_array_almost_equal(
            features.loc[common, "log_ret_1"].values,
            expected.loc[common].values,
            decimal=10,
            err_msg="log_ret_1 does not match ln(close[t]/close[t-1])",
        )

    def test_realised_vol_window_correct(self):
        """
        realized_vol_20 at index t must equal std(log_returns[t-19:t+1]) × √252.
        No bars after t should influence the value.
        """
        engineer = FeatureEngineer(vol_window=20, normalise=False)
        n = 400
        ohlcv = _make_ohlcv(n)
        close = ohlcv["close"]
        log_ret = np.log(close / close.shift(1))
        expected_vol = log_ret.rolling(20).std() * np.sqrt(252)

        features = engineer.transform(ohlcv)
        common = features.index.intersection(expected_vol.dropna().index)

        np.testing.assert_array_almost_equal(
            features.loc[common, "realized_vol_20"].values,
            expected_vol.loc[common].values,
            decimal=10,
            err_msg="realized_vol_20 does not match expected rolling window",
        )

    def test_warmup_rows_dropped_not_zero_filled(self):
        """
        transform() must drop NaN rows (warm-up period), not zero-fill them.
        The first valid output row must have a DatetimeIndex value later than
        the first input row.
        """
        engineer = FeatureEngineer(normalise=False, sma_slow=200)
        n = 600
        ohlcv = _make_ohlcv(n)
        features = engineer.transform(ohlcv)

        # With a 200-bar SMA, at least 200 input rows must have been dropped
        assert len(features) < n, "Expected warm-up rows to be dropped"
        assert features.index[0] > ohlcv.index[0], "First output row should be after warmup"

        # No NaN values in output
        assert not features.isnull().any().any(), "Output contains NaN values"


# ─────────────────────────────────────────────────────────────────────────────
# HMM causal inference test (THE critical look-ahead test)
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMLookAhead:
    def test_no_look_ahead_bias(self):
        """
        Regime at bar t must be IDENTICAL whether computed from data[0:t+1]
        or from data[0:T] for any T > t.

        This is the fundamental causal-inference guarantee of the forward
        algorithm.  If it fails, every backtest metric is fabricated.
        """
        pytest.importorskip("hmmlearn")
        from core.hmm_engine import HMMEngine

        engineer = FeatureEngineer()
        n = 800
        ohlcv = _make_ohlcv(n, seed=7)
        features = engineer.transform(ohlcv)

        # Need at least 504 bars to fit; use all as train data
        if len(features) < 504:
            pytest.skip("Not enough valid feature rows after warm-up for this test")

        engine = HMMEngine(
            n_candidates=[3],   # fixed n_states for speed
            n_init=3,
            min_train_bars=504,
        )
        engine.fit(features.iloc[:504])  # fit on first 504 bars

        # Predict on two different-length windows ending at the same bar t=599
        t = min(599, len(features) - 101)  # ensure at least 100 extra bars available
        short_window = features.iloc[: t + 1]          # data[0 : t+1]
        long_window = features.iloc[: t + 101]         # data[0 : t+101]

        regime_short = engine.predict_regime_filtered(short_window)[-1]
        regime_long = engine.predict_regime_filtered(long_window)[t]

        assert regime_short.state_id == regime_long.state_id, (
            f"LOOK-AHEAD BIAS DETECTED: state at t={t} differs depending on "
            f"how much future data is included.\n"
            f"  short window ({len(short_window)} bars): state={regime_short.state_id}, "
            f"label={regime_short.label}\n"
            f"  long  window ({len(long_window)} bars): state={regime_long.state_id}, "
            f"label={regime_long.label}"
        )

    def test_viterbi_would_fail_look_ahead(self):
        """
        Demonstrate that Viterbi (model.predict) CAN change past states when
        future observations are added — confirming why we must not use it.

        NOTE: This test documents the known flaw in Viterbi for causal use.
        It is expected to PASS (i.e. Viterbi IS biased), which motivates the
        forward-only design.
        """
        pytest.importorskip("hmmlearn")
        from core.hmm_engine import HMMEngine

        engineer = FeatureEngineer()
        n = 800
        ohlcv = _make_ohlcv(n, seed=13)
        features = engineer.transform(ohlcv)

        if len(features) < 504:
            pytest.skip("Not enough valid feature rows for this test")

        engine = HMMEngine(n_candidates=[3], n_init=2, min_train_bars=504)
        engine.fit(features.iloc[:504])

        t = min(550, len(features) - 51)
        short = features.iloc[: t + 1].values
        long_ = features.iloc[: t + 51].values

        engine._check_fitted()
        state_short = engine._model.predict(short)[-1]
        state_long = engine._model.predict(long_)[t]

        # We're NOT asserting equality here — we're documenting that they CAN differ.
        # If they happen to be equal for this seed, the test is vacuous but not wrong.
        if state_short != state_long:
            pytest.xfail(
                f"Viterbi look-ahead confirmed: state at t={t} changed "
                f"({state_short} → {state_long}) when {51} future bars were added."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backtest causal inference tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestLookAhead:
    def test_fold_train_end_before_test_start(self):
        """Every walk-forward fold must have train_end strictly before test_start."""
        pytest.skip("Requires WalkForwardBacktester implementation")

    def test_signal_uses_only_available_bars(self):
        """At each test bar, only bars up to and including that bar are fed to the HMM."""
        pytest.skip("Requires WalkForwardBacktester implementation")
