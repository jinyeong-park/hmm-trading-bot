"""
feature_engineering.py — Technical indicators and feature matrix construction.

All 14 features are computed from OHLCV data using only past and present
observations (no look-ahead). All features are standardised with rolling
z-scores (252-bar lookback, min_periods=126) before being fed to the HMM.

Feature set
-----------
  Returns:        log_ret_1, log_ret_5, log_ret_20
  Volatility:     realized_vol_20, vol_ratio (5d/20d)
  Volume:         volume_zscore (vs 50d), volume_trend (slope of 10d SMA)
  Trend:          adx_14, sma_50_slope
  Mean reversion: rsi_14_zscore, dist_sma_200
  Momentum:       roc_10, roc_20
  Range:          atr_14_pct
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ta  # type: ignore
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    ta = None  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical feature order (must stay in sync with HMMEngine expectations)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "log_ret_1",        # 1-period log return          ← feature 0 (used by HMM for return-labelling)
    "log_ret_5",        # 5-period log return
    "log_ret_20",       # 20-period log return
    "realized_vol_20",  # 20-bar rolling vol (annualised)
    "vol_ratio",        # 5-bar vol / 20-bar vol
    "volume_zscore",    # (volume − 50d_mean) / 50d_std
    "volume_trend",     # 1-bar pct-change of 10-bar volume SMA
    "adx_14",           # Average Directional Index, 14-period
    "sma_50_slope",     # 5-bar pct-change of 50-bar SMA ÷ 5  (per-bar slope)
    "rsi_14_zscore",    # RSI(14) after 252-bar rolling z-score
    "dist_sma_200",     # (close − SMA200) / SMA200
    "roc_10",           # (close − close[t-10]) / close[t-10]
    "roc_20",           # (close − close[t-20]) / close[t-20]
    "atr_14_pct",       # ATR(14) / close
]


class FeatureEngineer:
    """
    Builds the HMM feature matrix from raw OHLCV data.

    All computations are strictly causal: feature value at bar t depends only
    on OHLCV data at bars 0 … t.  The warm-up period (first N bars where any
    indicator has insufficient history) is dropped via dropna() at the end of
    transform().

    Parameters
    ----------
    vol_window : int
        Rolling window for realised volatility (default 20).
    vol_short_window : int
        Short window for vol-ratio numerator (default 5).
    vol_zscore_window : int
        Window for volume z-score baseline (default 50).
    vol_trend_window : int
        Window for volume SMA trend (default 10).
    atr_window : int
        ATR period (default 14).
    adx_window : int
        ADX period (default 14).
    rsi_window : int
        RSI period (default 14).
    sma_fast : int
        Fast SMA period used for slope (default 50).
    sma_slow : int
        Slow SMA period used for distance feature (default 200).
    roc_fast : int
        Fast ROC lookback (default 10).
    roc_slow : int
        Slow ROC lookback (default 20).
    normalise : bool
        If True, apply rolling z-score to all output features (default True).
    normalise_window : int
        Rolling window for z-score normalisation (default 252).
    normalise_min_periods : int
        Minimum periods to produce a valid z-score (default 126).
    """

    def __init__(
        self,
        vol_window: int = 20,
        vol_short_window: int = 5,
        vol_zscore_window: int = 50,
        vol_trend_window: int = 10,
        atr_window: int = 14,
        adx_window: int = 14,
        rsi_window: int = 14,
        sma_fast: int = 50,
        sma_slow: int = 200,
        roc_fast: int = 10,
        roc_slow: int = 20,
        normalise: bool = True,
        normalise_window: int = 252,
        normalise_min_periods: int = 126,
    ) -> None:
        if not _TA_AVAILABLE:
            raise ImportError(
                "The 'ta' library is required for FeatureEngineer. "
                "Install it with: pip install ta"
            )
        self.vol_window = vol_window
        self.vol_short_window = vol_short_window
        self.vol_zscore_window = vol_zscore_window
        self.vol_trend_window = vol_trend_window
        self.atr_window = atr_window
        self.adx_window = adx_window
        self.rsi_window = rsi_window
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.roc_fast = roc_fast
        self.roc_slow = roc_slow
        self.normalise = normalise
        self.normalise_window = normalise_window
        self.normalise_min_periods = normalise_min_periods

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full 14-feature matrix from raw OHLCV bars.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Must contain lowercase columns: open, high, low, close, volume.
            A DatetimeIndex is strongly recommended.

        Returns
        -------
        pd.DataFrame
            Shape (n_valid_bars, 14).  NaN warm-up rows are dropped.
            Column order matches FEATURE_NAMES.

        Raises
        ------
        ValueError
            If required columns are missing or the DataFrame is empty.
        """
        self._validate_ohlcv(ohlcv)

        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float)

        # ── 1–3: Returns ──────────────────────────────────────────────
        log_ret_1 = self._log_returns(close, 1)
        log_ret_5 = self._log_returns(close, 5)
        log_ret_20 = self._log_returns(close, 20)

        # ── 4–5: Volatility ───────────────────────────────────────────
        realized_vol_20 = self._realised_volatility(log_ret_1)
        vol_short = log_ret_1.rolling(self.vol_short_window).std()
        vol_long = log_ret_1.rolling(self.vol_window).std()
        vol_ratio = vol_short / (vol_long + 1e-10)

        # ── 6–7: Volume ───────────────────────────────────────────────
        vol_mean = volume.rolling(self.vol_zscore_window).mean()
        vol_std = volume.rolling(self.vol_zscore_window).std()
        volume_zscore = (volume - vol_mean) / (vol_std + 1e-10)

        vol_sma = volume.rolling(self.vol_trend_window).mean()
        volume_trend = vol_sma.pct_change(1)

        # ── 8: Trend — ADX ────────────────────────────────────────────
        adx_14 = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=self.adx_window, fillna=False
        ).adx()

        # ── 9: Trend — SMA slope ─────────────────────────────────────
        sma_fast = close.rolling(self.sma_fast).mean()
        # Per-bar normalised slope: 5-bar pct-change ÷ 5
        sma_50_slope = sma_fast.pct_change(5) / 5.0

        # ── 10: Mean reversion — RSI z-score ─────────────────────────
        rsi_raw = ta.momentum.RSIIndicator(
            close=close, window=self.rsi_window, fillna=False
        ).rsi()
        # Z-score RSI before the global normalisation pass (values 0–100 → centred)
        rsi_14_zscore = self._rolling_zscore(
            rsi_raw,
            window=self.normalise_window,
            min_periods=self.normalise_min_periods,
        )

        # ── 11: Mean reversion — distance from 200 SMA ───────────────
        sma_slow = close.rolling(self.sma_slow).mean()
        dist_sma_200 = (close - sma_slow) / (sma_slow + 1e-10)

        # ── 12–13: Momentum — ROC ─────────────────────────────────────
        roc_10 = (close - close.shift(self.roc_fast)) / (close.shift(self.roc_fast) + 1e-10)
        roc_20 = (close - close.shift(self.roc_slow)) / (close.shift(self.roc_slow) + 1e-10)

        # ── 14: Range — normalised ATR ────────────────────────────────
        atr_raw = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=self.atr_window, fillna=False
        ).average_true_range()
        atr_14_pct = atr_raw / (close + 1e-10)

        # ── Assemble ──────────────────────────────────────────────────
        raw = pd.DataFrame(
            {
                "log_ret_1": log_ret_1,
                "log_ret_5": log_ret_5,
                "log_ret_20": log_ret_20,
                "realized_vol_20": realized_vol_20,
                "vol_ratio": vol_ratio,
                "volume_zscore": volume_zscore,
                "volume_trend": volume_trend,
                "adx_14": adx_14,
                "sma_50_slope": sma_50_slope,
                "rsi_14_zscore": rsi_14_zscore,
                "dist_sma_200": dist_sma_200,
                "roc_10": roc_10,
                "roc_20": roc_20,
                "atr_14_pct": atr_14_pct,
            },
            index=ohlcv.index,
        )

        # ── Global z-score normalisation (skip rsi which is pre-normalised) ──
        if self.normalise:
            cols_to_norm = [c for c in raw.columns if c != "rsi_14_zscore"]
            raw[cols_to_norm] = self._zscore_normalise(raw[cols_to_norm])

        # ── Drop warm-up NaN rows ─────────────────────────────────────
        out = raw.dropna()

        logger.debug(
            "FeatureEngineer.transform: %d bars in → %d valid features out "
            "(dropped %d NaN warm-up rows)",
            len(ohlcv),
            len(out),
            len(ohlcv) - len(out),
        )
        return out

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names produced by transform()."""
        return FEATURE_NAMES.copy()

    # ─────────────────────────────────────────────────────────────────
    # Individual feature helpers (also exposed for unit tests)
    # ─────────────────────────────────────────────────────────────────

    def _log_returns(self, close: pd.Series, period: int = 1) -> pd.Series:
        """log(close[t] / close[t − period])."""
        return np.log(close / close.shift(period))

    def _realised_volatility(self, log_returns: pd.Series) -> pd.Series:
        """Annualised realised vol = rolling_std(vol_window) × √252."""
        return log_returns.rolling(self.vol_window).std() * np.sqrt(252)

    def _atr_pct(self, ohlcv: pd.DataFrame) -> pd.Series:
        """ATR(atr_window) / close."""
        atr = ta.volatility.AverageTrueRange(
            high=ohlcv["high"].astype(float),
            low=ohlcv["low"].astype(float),
            close=ohlcv["close"].astype(float),
            window=self.atr_window,
            fillna=False,
        ).average_true_range()
        return atr / (ohlcv["close"].astype(float) + 1e-10)

    def _rsi(self, close: pd.Series) -> pd.Series:
        """RSI(rsi_window), values in [0, 100]."""
        return ta.momentum.RSIIndicator(
            close=close.astype(float), window=self.rsi_window, fillna=False
        ).rsi()

    def _macd_signal(self, close: pd.Series) -> pd.Series:
        """MACD histogram (MACD line − signal line, default 12/26/9)."""
        return ta.trend.MACD(close=close.astype(float)).macd_diff()

    def _bollinger_pct(self, close: pd.Series) -> pd.Series:
        """Bollinger Band %B: (price − lower) / (upper − lower)."""
        return ta.volatility.BollingerBands(close=close.astype(float)).bollinger_pband()

    # ─────────────────────────────────────────────────────────────────
    # Normalisation helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_zscore(
        series: pd.Series,
        window: int = 252,
        min_periods: int = 126,
    ) -> pd.Series:
        """
        Per-bar rolling z-score: (x − rolling_mean) / rolling_std.

        Causal by construction: at bar t, only bars 0 … t contribute.

        Parameters
        ----------
        series : pd.Series
        window : int
        min_periods : int

        Returns
        -------
        pd.Series
        """
        mean = series.rolling(window=window, min_periods=min_periods).mean()
        std = series.rolling(window=window, min_periods=min_periods).std()
        return (series - mean) / (std + 1e-10)

    def _zscore_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score to every column of df (in-place copy)."""
        result = df.copy()
        for col in df.columns:
            result[col] = self._rolling_zscore(
                df[col],
                window=self.normalise_window,
                min_periods=self.normalise_min_periods,
            )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
        """Raise ValueError if ohlcv is missing required columns or is empty."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(
                f"OHLCV DataFrame is missing required columns: {missing}. "
                f"Found: {list(ohlcv.columns)}"
            )
        if len(ohlcv) == 0:
            raise ValueError("OHLCV DataFrame is empty.")
