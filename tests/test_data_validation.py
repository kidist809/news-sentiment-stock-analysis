import pandas as pd
import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validation import validate_news, validate_stock, check_date_overlap


def make_news():
    return pd.DataFrame({
        'headline': ['Stock hits high', 'Earnings beat', None],
        'date':     pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        'stock':    ['AAPL', 'GOOG', 'AAPL'],
    })


def make_stock():
    idx = pd.date_range('2020-01-01', periods=5, freq='B')
    return pd.DataFrame({'Close': [100, 101, 102, 103, 104],
                         'Adj Close': [100, 101, 102, 103, 104]}, index=idx)


def test_validate_news_counts():
    result = validate_news(make_news())
    assert result['total_rows'] == 3
    assert result['missing_headline'] == 1
    assert result['unique_stocks'] == 2


def test_validate_news_no_missing():
    df = make_news().dropna()
    result = validate_news(df)
    assert result['missing_headline'] == 0


def test_validate_stock_counts():
    result = validate_stock(make_stock(), 'AAPL')
    assert result['ticker'] == 'AAPL'
    assert result['total_rows'] == 5
    assert result['missing_close'] == 0


def test_check_date_overlap_full():
    dates = [datetime.date(2020, 1, i) for i in range(1, 6)]
    result = check_date_overlap(dates, dates)
    assert result['overlap_days'] == 5
    assert result['coverage_pct'] == 100.0


def test_check_date_overlap_partial():
    news_dates  = [datetime.date(2020, 1, i) for i in range(1, 6)]
    price_dates = [datetime.date(2020, 1, i) for i in range(3, 8)]
    result = check_date_overlap(news_dates, price_dates)
    assert result['overlap_days'] == 3
    assert result['coverage_pct'] == 60.0


def test_check_date_overlap_empty():
    result = check_date_overlap([], [])
    assert result['coverage_pct'] == 0
