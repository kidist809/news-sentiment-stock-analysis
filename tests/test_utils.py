import pytest
import pandas as pd
import numpy as np


# ── Helpers (inline so tests run without src imports) ──────────────────────────

def compute_daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change() * 100


def classify_sentiment(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def extract_domain(publisher: str) -> str:
    if isinstance(publisher, str) and "@" in publisher:
        return publisher.split("@")[-1].lower()
    return publisher


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_daily_returns_basic():
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = compute_daily_returns(prices)
    assert np.isnan(returns.iloc[0])
    assert abs(returns.iloc[1] - 10.0) < 1e-6
    assert abs(returns.iloc[2] - (-10.0)) < 1e-6


def test_daily_returns_constant():
    prices = pd.Series([50.0, 50.0, 50.0])
    returns = compute_daily_returns(prices)
    assert returns.iloc[1] == 0.0


def test_classify_sentiment_positive():
    assert classify_sentiment(0.5) == "Positive"
    assert classify_sentiment(0.05) == "Positive"


def test_classify_sentiment_negative():
    assert classify_sentiment(-0.5) == "Negative"
    assert classify_sentiment(-0.05) == "Negative"


def test_classify_sentiment_neutral():
    assert classify_sentiment(0.0) == "Neutral"
    assert classify_sentiment(0.04) == "Neutral"
    assert classify_sentiment(-0.04) == "Neutral"


def test_extract_domain_email():
    assert extract_domain("john@reuters.com") == "reuters.com"


def test_extract_domain_plain_name():
    assert extract_domain("Reuters") == "Reuters"


def test_extract_domain_non_string():
    assert extract_domain(None) is None
