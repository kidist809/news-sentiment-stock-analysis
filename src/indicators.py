"""
Technical indicator utilities — pure pandas, no TA-Lib required.
"""
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram. Returns (macd, signal, hist)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def add_indicators(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """Add SMA(20/50), EMA(20), RSI(14), MACD(12,26,9) and return columns to df."""
    close = df[price_col]
    df['SMA_20'] = sma(close, 20)
    df['SMA_50'] = sma(close, 50)
    df['EMA_20'] = ema(close, 20)
    df['RSI_14'] = rsi(close, 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(close)
    return df


def pynance_metrics(df: pd.DataFrame, price_col: str = 'Adj Close') -> dict:
    """
    PyNance-style financial metrics:
    daily returns, rolling volatility, Sharpe ratio, total return.
    """
    ret = df[price_col].pct_change().dropna()
    df['daily_return'] = df[price_col].pct_change() * 100
    df['vol_20'] = df['daily_return'].rolling(20).std()
    df['vol_50'] = df['daily_return'].rolling(50).std()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    total_return = (df[price_col].iloc[-1] / df[price_col].iloc[0] - 1) * 100
    return {
        'mean_daily_return_pct': round(ret.mean() * 100, 4),
        'volatility_pct':        round(ret.std() * 100, 4),
        'sharpe_ratio':          round(sharpe, 3),
        'total_return_pct':      round(total_return, 2),
    }
