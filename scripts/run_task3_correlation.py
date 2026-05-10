"""
Task 3 - Correlation: News Sentiment vs. Stock Returns
Run: venv/Scripts/python scripts/run_task3_correlation.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
import yfinance as yf
import os

nltk.download('vader_lexicon', quiet=True)
plt.style.use('seaborn-v0_8-whitegrid')

OUT = 'data/raw'

# Top tickers with strong news coverage (from Task 1 analysis)
TICKERS = ['NVDA', 'TSLA', 'NFLX', 'QCOM', 'ORCL', 'BABA', 'GOOG', 'AAPL', 'AMZN', 'FB']
TICKER_MAP = {'FB': 'META'}   # FB in news -> META for price download


# ── 1. Load news ──────────────────────────────────────────────────────────────
print("Loading news data...")
news = pd.read_csv(f'{OUT}/newsData/raw_analyst_ratings.csv', index_col=0)
news['date'] = pd.to_datetime(news['date'], utc=True, errors='coerce')
news.dropna(subset=['date', 'headline', 'stock'], inplace=True)
news['trade_date'] = news['date'].dt.tz_localize(None).dt.normalize().dt.date
news = news[news['stock'].isin(TICKERS)]
print(f"News rows: {len(news)}")
print(news['stock'].value_counts().to_string())


# ── 2. Download / load stock prices aligned to news date range ────────────────
news_start = str(news['date'].min().date())
news_end   = str(news['date'].max().date())
print(f"\nNews date range: {news_start} to {news_end}")

price_tickers = [TICKER_MAP.get(t, t) for t in TICKERS]

def load_or_download(ticker, start, end):
    path = f'{OUT}/{ticker}_task3.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            return None
        # flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Adj Close'], inplace=True)
    df.sort_index(inplace=True)
    return df

print("\nLoading stock prices...")
price_frames = []
for news_t, price_t in zip(TICKERS, price_tickers):
    d = load_or_download(price_t, news_start, news_end)
    if d is None or len(d) < 10:
        print(f"  {price_t}: skipped (no data)")
        continue
    d['daily_return'] = d['Adj Close'].pct_change() * 100
    d['stock'] = news_t   # keep news ticker for merging
    d.index = pd.to_datetime(d.index).date
    price_frames.append(d[['daily_return', 'stock']].dropna())
    print(f"  {price_t}: {len(d)} rows")

prices = pd.concat(price_frames).reset_index().rename(columns={'index': 'trade_date'})
print(f"Total price rows: {len(prices)}")


# ── 3. VADER sentiment scoring ────────────────────────────────────────────────
print("\nScoring sentiment with VADER...")
sia = SentimentIntensityAnalyzer()
news['sentiment'] = news['headline'].apply(
    lambda h: sia.polarity_scores(str(h))['compound']
)
print(f"Sentiment stats:\n{news['sentiment'].describe().round(4)}")

daily_sentiment = (
    news.groupby(['stock', 'trade_date'])['sentiment']
    .mean()
    .reset_index()
    .rename(columns={'sentiment': 'avg_sentiment'})
)
print(f"Daily sentiment rows: {len(daily_sentiment)}")


# ── 4. Merge ──────────────────────────────────────────────────────────────────
merged = pd.merge(daily_sentiment, prices, on=['stock', 'trade_date'], how='inner')
print(f"Merged rows: {len(merged)}")
print(merged['stock'].value_counts().to_string())

if len(merged) < 5:
    print("ERROR: Not enough merged rows for correlation. Check date alignment.")
    exit(1)


# ── 5. Pearson correlation ────────────────────────────────────────────────────
r_all, p_all = stats.pearsonr(merged['avg_sentiment'], merged['daily_return'])
print(f"\nOverall Pearson r = {r_all:.4f}  (p = {p_all:.4f})")

def safe_pearson(g):
    if len(g) < 3:
        return pd.Series({'pearson_r': np.nan, 'p_value': np.nan, 'n': len(g)})
    r, p = stats.pearsonr(g['avg_sentiment'], g['daily_return'])
    return pd.Series({'pearson_r': r, 'p_value': p, 'n': len(g)})

corr_by_stock = (
    merged.groupby('stock')
    .apply(safe_pearson)
    .reset_index()
    .dropna(subset=['pearson_r'])
)
print("\n=== Per-Stock Correlation ===")
print(corr_by_stock.round(4).to_string(index=False))


# ── 6. Scatter plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(merged['avg_sentiment'], merged['daily_return'],
           alpha=0.3, s=15, color='steelblue')
m, b = np.polyfit(merged['avg_sentiment'], merged['daily_return'], 1)
x_line = np.linspace(merged['avg_sentiment'].min(), merged['avg_sentiment'].max(), 100)
ax.plot(x_line, m * x_line + b, color='red', lw=1.5,
        label=f'r = {r_all:.3f}   p = {p_all:.3f}')
ax.set_xlabel('Average Daily Sentiment Score (VADER compound)')
ax.set_ylabel('Daily Return (%)')
ax.set_title('News Sentiment vs. Daily Stock Return')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_sentiment_vs_return.png', dpi=150)
plt.close()
print("\nSaved fig10_sentiment_vs_return.png")


# ── 7. Per-stock correlation bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = ['green' if v > 0 else 'red' for v in corr_by_stock['pearson_r']]
ax.bar(corr_by_stock['stock'], corr_by_stock['pearson_r'], color=colors, alpha=0.75)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Pearson Correlation: Sentiment vs. Return by Stock')
ax.set_ylabel('Pearson r')
ax.set_xlabel('Stock')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_correlation_by_stock.png', dpi=150)
plt.close()
print("Saved fig11_correlation_by_stock.png")


# ── 8. Average return by sentiment category ───────────────────────────────────
def classify(s):
    if s >= 0.05:    return 'Positive'
    elif s <= -0.05: return 'Negative'
    return 'Neutral'

merged['sentiment_class'] = merged['avg_sentiment'].apply(classify)
avg_by_class = merged.groupby('sentiment_class')['daily_return'].mean()
counts = merged['sentiment_class'].value_counts()

print("\n=== Avg Daily Return by Sentiment Class ===")
for cls in ['Positive', 'Neutral', 'Negative']:
    print(f"  {cls}: {avg_by_class.get(cls, 0):.4f}%  (n={counts.get(cls, 0)})")

fig, ax = plt.subplots(figsize=(6, 4))
order = ['Positive', 'Neutral', 'Negative']
color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
vals = [avg_by_class.get(c, 0) for c in order]
ax.bar(order, vals, color=[color_map[c] for c in order], alpha=0.75)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Average Daily Return by Sentiment Category')
ax.set_ylabel('Avg Return (%)')
ax.set_xlabel('Sentiment Category')
plt.tight_layout()
plt.savefig(f'{OUT}/fig12_return_by_sentiment_class.png', dpi=150)
plt.close()
print("Saved fig12_return_by_sentiment_class.png")


# ── 9. Lag-1 analysis ────────────────────────────────────────────────────────
print("\nRunning lag-1 correlation analysis...")
lag_results = []
for t in merged['stock'].unique():
    sub = merged[merged['stock'] == t].sort_values('trade_date').copy()
    sub['next_day_return'] = sub['daily_return'].shift(-1)
    sub.dropna(inplace=True)
    if len(sub) < 5:
        continue
    r_lag, p_lag = stats.pearsonr(sub['avg_sentiment'], sub['next_day_return'])
    lag_results.append({'stock': t, 'lag1_r': round(r_lag, 4), 'lag1_p': round(p_lag, 4)})

if lag_results:
    lag_df = pd.DataFrame(lag_results)
    print("=== Lag-1 Correlation (sentiment -> next day return) ===")
    print(lag_df.to_string(index=False))

    common = corr_by_stock.set_index('stock')
    lag_df = lag_df[lag_df['stock'].isin(common.index)].reset_index(drop=True)

    if len(lag_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(lag_df))
        w = 0.35
        same_day_r = common.loc[lag_df['stock'], 'pearson_r'].values
        ax.bar(x - w/2, same_day_r,          w, label='Same-day',  color='steelblue', alpha=0.8)
        ax.bar(x + w/2, lag_df['lag1_r'].values, w, label='Lag-1 day', color='coral', alpha=0.8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lag_df['stock'], rotation=45)
        ax.set_title('Same-day vs. Lag-1 Sentiment Correlation')
        ax.set_ylabel('Pearson r')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{OUT}/fig13_lag_correlation.png', dpi=150)
        plt.close()
        print("Saved fig13_lag_correlation.png")
else:
    print("Skipping lag chart - insufficient data per stock")

print("\nTask 3 complete. All figures saved to data/raw/")
