"""
Data validation utilities for news and stock price datasets.
"""
import pandas as pd


def validate_news(df: pd.DataFrame) -> dict:
    """Return basic quality stats for the news DataFrame."""
    return {
        'total_rows':       len(df),
        'missing_headline': df['headline'].isnull().sum(),
        'missing_date':     df['date'].isnull().sum(),
        'missing_stock':    df['stock'].isnull().sum(),
        'unique_stocks':    df['stock'].nunique(),
        'date_min':         str(df['date'].min()),
        'date_max':         str(df['date'].max()),
    }


def validate_stock(df: pd.DataFrame, ticker: str) -> dict:
    """Return basic quality stats for a stock price DataFrame."""
    return {
        'ticker':        ticker,
        'total_rows':    len(df),
        'missing_close': df['Close'].isnull().sum() if 'Close' in df.columns else 'N/A',
        'date_min':      str(df.index.min()),
        'date_max':      str(df.index.max()),
    }


def check_date_overlap(news_dates, price_dates) -> dict:
    """Check how many news dates have a matching trading day."""
    news_set  = set(news_dates)
    price_set = set(price_dates)
    overlap   = news_set & price_set
    return {
        'news_days':    len(news_set),
        'price_days':   len(price_set),
        'overlap_days': len(overlap),
        'coverage_pct': round(len(overlap) / len(news_set) * 100, 2) if news_set else 0,
    }
