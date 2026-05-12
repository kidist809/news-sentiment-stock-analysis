"""
Task 3 — Correlation: News Sentiment vs. Stock Returns
Run: venv/Scripts/python scripts/run_task3_correlation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
import yfinance as yf

from sentiment import classify_sentiment

nltk.download('vader_lexicon', quiet=True)
plt.style.use('seaborn-v0_8-whitegrid')

OUT = 'data/raw'

# ── 1. Load news and find top tickers by coverage ────────────────────────────
print("Loading news data...")
news = pd.read_csv(f'{OUT}/newsData/raw_analyst_ratings.csv', index_col=0)
news['date'] = pd.to_datetime(news['date'], utc=True, errors='coerce')
news.dropna(subset=['date', 'headline', 'stock'], inplace=True)
news['trade_date'] = news['date'].dt.tz_localize(None).dt.normalize().dt.date

# Use top 30 tickers by article count to maximise matched rows
top_tickers = news['stock'].value_counts().head(30).index.tolist()
news = news[news['stock'].isin(top_tickers)]
print(f"News rows (top 30 tickers): {len(news)}")
print(news['stock'].value_counts().head(10).to_string())

news_start = str(news['date'].min().date())
news_end   = str(news['date'].max().date())
print(f"\nNews date range: {news_start} to {news_end}")

# ── 2. Download / load stock prices ──────────────────────────────────────────
# Known ticker renames: map news ticker -> yfinance ticker
TICKER_MAP = {'FB': 'META', 'BRKB': 'BRK-B'}

def load_or_download(news_ticker, start, end):
    yf_ticker = TICKER_MAP.get(news_ticker, news_ticker)
    path = f'{OUT}/{yf_ticker}_task3.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            return None
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
for t in top_tickers:
    d = load_or_download(t, news_start, news_end)
    if d is None or len(d) < 10:
        print(f"  {t}: skipped")
        continue
    d['daily_return'] = d['Adj Close'].pct_change() * 100
    d['stock'] = t
    d.index = pd.to_datetime(d.index).date
    price_frames.append(d[['daily_return', 'stock']].dropna())
    print(f"  {t}: {len(d)} rows")

prices = pd.concat(price_frames).reset_index().rename(columns={'index': 'trade_date'})
print(f"Total price rows: {len(prices)}")

# ── 3. VADER sentiment scoring ────────────────────────────────────────────────
print("\nScoring sentiment with VADER...")
sia = SentimentIntensityAnalyzer()
news['sentiment'] = news['headline'].apply(
    lambda h: sia.polarity_scores(str(h))['compound']
)
news['sentiment_class'] = news['sentiment'].apply(classify_sentiment)
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
print(f"\nMerged rows: {len(merged)}")
print(merged['stock'].value_counts().to_string())

if len(merged) < 5:
    print("ERROR: Not enough matched rows. Check date alignment.")
    sys.exit(1)

# ── 5. Overall Pearson correlation ───────────────────────────────────────────
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
    .sort_values('pearson_r', ascending=False)
)
print("\n=== Per-Stock Correlation ===")
print(corr_by_stock.round(4).to_string(index=False))

# ── 6. Scatter: sentiment vs return ──────────────────────────────────────────
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
top_corr = corr_by_stock[corr_by_stock['n'] >= 5].head(15)
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['green' if v > 0 else 'red' for v in top_corr['pearson_r']]
ax.bar(top_corr['stock'], top_corr['pearson_r'], color=colors, alpha=0.75)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Pearson r: Sentiment vs. Same-day Return (stocks with n ≥ 5 days)')
ax.set_ylabel('Pearson r')
ax.set_xlabel('Stock')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_correlation_by_stock.png', dpi=150)
plt.close()
print("Saved fig11_correlation_by_stock.png")

# ── 8. Return by sentiment category ──────────────────────────────────────────
avg_by_class = merged.groupby('sentiment_class')['daily_return'].mean()
counts = merged['sentiment_class'].value_counts()
print("\n=== Avg Daily Return by Sentiment Class ===")
for cls in ['Positive', 'Neutral', 'Negative']:
    print(f"  {cls}: {avg_by_class.get(cls, 0):.4f}%  (n={counts.get(cls, 0)})")

fig, ax = plt.subplots(figsize=(6, 4))
order = ['Positive', 'Neutral', 'Negative']
color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
ax.bar(order, [avg_by_class.get(c, 0) for c in order],
       color=[color_map[c] for c in order], alpha=0.75)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Average Daily Return by Sentiment Category')
ax.set_ylabel('Avg Return (%)')
ax.set_xlabel('Sentiment Category')
plt.tight_layout()
plt.savefig(f'{OUT}/fig12_return_by_sentiment_class.png', dpi=150)
plt.close()
print("Saved fig12_return_by_sentiment_class.png")

# ── 9. Lag-0 vs Lag-1 vs Lag-2 correlation ───────────────────────────────────
print("\nRunning multi-lag correlation analysis...")
lag_results = []
for t in merged['stock'].unique():
    sub = merged[merged['stock'] == t].sort_values('trade_date').copy()
    if len(sub) < 5:
        continue
    row = {'stock': t, 'n': len(sub)}
    for lag in range(3):
        sub[f'ret_lag{lag}'] = sub['daily_return'].shift(-lag)
    sub.dropna(inplace=True)
    if len(sub) < 3:
        continue
    for lag in range(3):
        r_lag, _ = stats.pearsonr(sub['avg_sentiment'], sub[f'ret_lag{lag}'])
        row[f'lag{lag}_r'] = round(r_lag, 4)
    lag_results.append(row)

if lag_results:
    lag_df = pd.DataFrame(lag_results).set_index('stock')
    print(lag_df.to_string())

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(lag_df))
    w = 0.25
    colors_lag = ['steelblue', 'coral', 'mediumseagreen']
    for i, lag in enumerate(range(3)):
        col = f'lag{lag}_r'
        if col in lag_df.columns:
            ax.bar(x + i * w, lag_df[col].values, w,
                   label=f'Lag-{lag}', color=colors_lag[i], alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(lag_df.index, rotation=45)
    ax.set_title('Sentiment Correlation: Lag-0, Lag-1, Lag-2')
    ax.set_ylabel('Pearson r')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig13_lag_correlation.png', dpi=150)
    plt.close()
    print("Saved fig13_lag_correlation.png")

print("\nTask 3 complete.")
