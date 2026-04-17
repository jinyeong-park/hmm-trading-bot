"""
test_look_ahead.py — Look-ahead bias prevention tests.

Verifies that no feature or signal uses future information:
  - Feature values at time t depend only on data up to and including t.
  - HMM Viterbi decode does not bleed future states backwards.
  - Backtest signal generation uses only bars available at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import FeatureEngineer


class TestFeatureLookAhead:
    def test_features_do_not_use_future_close(self):
        """
        Shuffle future prices and assert that past feature values are unchanged.
        """
        pytest.skip("Requires implementation")

    def test_log_return_uses_only_past_close(self):
        """
        Verify that log_return at index t equals ln(close[t] / close[t-1])
        and does not depend on close[t+1:].
        """
        pytest.skip("Requires implementation")

    def test_realised_vol_window_correct(self):
        """
        Assert that realised_vol at index t is computed from [t-window+1 : t] only.
        """
        pytest.skip("Requires implementation")

    def test_indicators_nan_at_warmup_period(self):
        """
        Indicators must produce NaN for the first (window - 1) bars, not look-ahead values.
        """
        engineer = FeatureEngineer()
        n = 300
        ohlcv = pd.DataFrame(
            {
                "open": np.random.rand(n) + 100,
                "high": np.random.rand(n) + 101,
                "low": np.random.rand(n) + 99,
                "close": np.random.rand(n) + 100,
                "volume": np.random.randint(1_000_000, 5_000_000, n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        with pytest.raises(NotImplementedError):
            engineer.transform(ohlcv)


class TestHMMLookAhead:
    def test_viterbi_decode_is_causal(self):
        """
        Verify that changing bar t+k does not alter the decoded state at bar t.
        (The Viterbi path is causal by construction, but verify empirically.)
        """
        pytest.skip("Requires implementation")


class TestBacktestLookAhead:
    def test_fold_train_end_before_test_start(self):
        """
        Every walk-forward fold must have train_end strictly before test_start.
        """
        pytest.skip("Requires implementation")

    def test_signal_uses_only_available_bars(self):
        """
        At each test bar, only bars up to and including that bar are fed to the HMM.
        """
        pytest.skip("Requires implementation")
