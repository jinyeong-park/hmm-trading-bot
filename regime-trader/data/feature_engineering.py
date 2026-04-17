"""
feature_engineering.py — Technical indicators and feature matrix construction.

Responsibilities:
  - Compute returns, log-returns, and realised volatility series.
  - Add technical indicators (ATR, RSI, MACD, Bollinger Bands) via the `ta` library.
  - Normalise/standardise features to stabilise HMM training.
  - Assemble the final feature DataFrame consumed by HMMEngine.
  - Ensure strict look-ahead prevention (all indicators use only past data).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import ta


class FeatureEngineer:
    """
    Builds the feature matrix used by HMMEngine from raw OHLCV data.

    The default feature set is:
      - log_return        : Log daily return
      - realised_vol_20   : 20-bar rolling realised volatility (annualised)
      - atr_14_pct        : 14-bar ATR normalised by close price
      - rsi_14            : 14-bar RSI
      - macd_signal       : MACD histogram (12/26/9)
      - bb_pct            : Bollinger Band %B (20-bar, 2σ)

    Parameters
    ----------
    vol_window : int
        Rolling window for realised volatility calculation.
    atr_window : int
        ATR period.
    rsi_window : int
        RSI period.
    normalise : bool
        If True, z-score normalise all features before returning.
    normalise_window : int
        Rolling window for z-score normalisation (use 0 for global).
    """

    def __init__(
        self,
        vol_window: int = 20,
        atr_window: int = 14,
        rsi_window: int = 14,
        normalise: bool = True,
        normalise_window: int = 252,
    ) -> None:
        self.vol_window = vol_window
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.normalise = normalise
        self.normalise_window = normalise_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full feature matrix from raw OHLCV bars.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Must contain columns: [open, high, low, close, volume].
            DatetimeIndex required.

        Returns
        -------
        pd.DataFrame
            Feature matrix with NaN rows dropped.
            Shape: (n_valid_bars, n_features).
        """
        raise NotImplementedError

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names produced by transform()."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Feature computations
    # ------------------------------------------------------------------

    def _log_returns(self, close: pd.Series) -> pd.Series:
        """
        Compute log returns: ln(close_t / close_{t-1}).

        Parameters
        ----------
        close : pd.Series

        Returns
        -------
        pd.Series
        """
        raise NotImplementedError

    def _realised_volatility(self, log_returns: pd.Series) -> pd.Series:
        """
        Compute rolling annualised realised volatility.

        Parameters
        ----------
        log_returns : pd.Series

        Returns
        -------
        pd.Series
            Annualised vol = rolling_std * sqrt(252)
        """
        raise NotImplementedError

    def _atr_pct(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Compute ATR as a percentage of close price.

        Parameters
        ----------
        ohlcv : pd.DataFrame

        Returns
        -------
        pd.Series
        """
        raise NotImplementedError

    def _rsi(self, close: pd.Series) -> pd.Series:
        """
        Compute RSI using the `ta` library.

        Parameters
        ----------
        close : pd.Series

        Returns
        -------
        pd.Series
            Values in [0, 100].
        """
        raise NotImplementedError

    def _macd_signal(self, close: pd.Series) -> pd.Series:
        """
        Compute MACD histogram (MACD line minus signal line).

        Parameters
        ----------
        close : pd.Series

        Returns
        -------
        pd.Series
        """
        raise NotImplementedError

    def _bollinger_pct(self, close: pd.Series) -> pd.Series:
        """
        Compute Bollinger Band %B: (price - lower) / (upper - lower).

        Parameters
        ----------
        close : pd.Series

        Returns
        -------
        pd.Series
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _zscore_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling z-score normalisation to each feature column.

        Uses a window of normalise_window bars. If normalise_window == 0,
        applies global (expanding) normalisation.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError
