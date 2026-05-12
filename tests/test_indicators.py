import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indicators import sma, ema, rsi, macd, add_indicators, pynance_metrics


@pytest.fixture
def price_series():
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
    return prices


@pytest.fixture
def stock_df(price_series):
    df = pd.DataFrame({'Close': price_series, 'Adj Close': price_series})
    return df


# ── SMA ───────────────────────────────────────────────────────────────────────

def test_sma_length(price_series):
    result = sma(price_series, 20)
    assert len(result) == len(price_series)


def test_sma_first_values_nan(price_series):
    result = sma(price_series, 20)
    assert result.iloc[:19].isna().all()
    assert not pd.isna(result.iloc[19])


def test_sma_constant_series():
    s = pd.Series([5.0] * 50)
    result = sma(s, 10)
    assert result.dropna().eq(5.0).all()


# ── EMA ───────────────────────────────────────────────────────────────────────

def test_ema_length(price_series):
    result = ema(price_series, 20)
    assert len(result) == len(price_series)


def test_ema_no_nan_after_first(price_series):
    result = ema(price_series, 20)
    # EMA with adjust=False produces values from index 0
    assert result.isna().sum() == 0


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_range(price_series):
    result = rsi(price_series, 14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_rsi_length(price_series):
    result = rsi(price_series, 14)
    assert len(result) == len(price_series)


# ── MACD ──────────────────────────────────────────────────────────────────────

def test_macd_returns_three_series(price_series):
    macd_line, signal_line, hist = macd(price_series)
    assert len(macd_line) == len(price_series)
    assert len(signal_line) == len(price_series)
    assert len(hist) == len(price_series)


def test_macd_hist_equals_diff(price_series):
    macd_line, signal_line, hist = macd(price_series)
    pd.testing.assert_series_equal(hist, macd_line - signal_line)


# ── add_indicators ────────────────────────────────────────────────────────────

def test_add_indicators_columns(stock_df):
    result = add_indicators(stock_df)
    for col in ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist']:
        assert col in result.columns


# ── pynance_metrics ───────────────────────────────────────────────────────────

def test_pynance_metrics_keys(stock_df):
    metrics = pynance_metrics(stock_df)
    for key in ['mean_daily_return_pct', 'volatility_pct', 'sharpe_ratio', 'total_return_pct']:
        assert key in metrics


def test_pynance_metrics_types(stock_df):
    metrics = pynance_metrics(stock_df)
    for v in metrics.values():
        assert isinstance(v, float)
