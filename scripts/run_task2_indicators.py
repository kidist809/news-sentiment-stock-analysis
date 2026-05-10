"""
Task 2 - Quantitative Analysis with Technical Indicators
Run: venv/Scripts/python scripts/run_task2_indicators.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

TICKERS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
OUT = 'data/raw'


def load_stock(ticker):
    """Load yfinance CSV with 3-row header (Price, Ticker, Date)."""
    df = pd.read_csv(f'{OUT}/{ticker}_historical.csv', header=[0, 1], index_col=0)
    # Flatten multi-level columns: keep only first level (Price type)
    df.columns = df.columns.get_level_values(0)
    # Drop the 'Ticker' row that yfinance inserts
    df = df[~df.index.isin(['Ticker', 'Date', ''])]
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Close', 'Adj Close'], inplace=True)
    df.sort_index(inplace=True)
    return df


# ── Indicator helpers (pandas fallback — no TA-Lib required) ──────────────────

def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ── 1. Load & compute indicators for AAPL ────────────────────────────────────
print("Loading AAPL...")
df = load_stock('AAPL')
print(f"AAPL shape: {df.shape}")
print(df[['Close', 'Adj Close', 'Volume']].tail(3))

close = df['Close']
df['SMA_20']       = sma(close, 20)
df['SMA_50']       = sma(close, 50)
df['EMA_20']       = ema(close, 20)
df['RSI_14']       = rsi(close, 14)
df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(close)
df['daily_return'] = df['Adj Close'].pct_change() * 100
df['vol_20']       = df['daily_return'].rolling(20).std()

print("\nSample indicators (last 5 rows):")
print(df[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD']].tail())


# ── 2. Plot: Price + MAs + RSI + MACD ────────────────────────────────────────
plot_df = df[df.index >= df.index.max() - pd.DateOffset(days=365)]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1, 1]})

# Panel 1 — Price + Moving Averages
axes[0].plot(plot_df.index, plot_df['Close'],   label='Close',  lw=1.2, color='black')
axes[0].plot(plot_df.index, plot_df['SMA_20'],  label='SMA 20', lw=1,   color='blue',   ls='--')
axes[0].plot(plot_df.index, plot_df['SMA_50'],  label='SMA 50', lw=1,   color='orange', ls='--')
axes[0].plot(plot_df.index, plot_df['EMA_20'],  label='EMA 20', lw=1,   color='green',  ls=':')
axes[0].set_title('AAPL - Price & Moving Averages')
axes[0].set_ylabel('Price (USD)')
axes[0].legend(fontsize=9)

# Panel 2 — RSI
axes[1].plot(plot_df.index, plot_df['RSI_14'], color='purple', lw=1)
axes[1].axhline(70, color='red',   ls='--', lw=0.8, label='Overbought (70)')
axes[1].axhline(30, color='green', ls='--', lw=0.8, label='Oversold (30)')
axes[1].set_ylabel('RSI (14)')
axes[1].set_ylim(0, 100)
axes[1].legend(fontsize=8)

# Panel 3 — MACD
axes[2].plot(plot_df.index, plot_df['MACD'],        label='MACD',   color='blue',   lw=1)
axes[2].plot(plot_df.index, plot_df['MACD_signal'], label='Signal', color='orange', lw=1)
axes[2].bar(plot_df.index,  plot_df['MACD_hist'],   label='Hist',   color='gray',   alpha=0.4, width=1)
axes[2].axhline(0, color='black', lw=0.5)
axes[2].set_ylabel('MACD')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_technical_indicators.png', dpi=150)
plt.close()
print("\nSaved fig7_technical_indicators.png")


# ── 3. Returns & Volatility plot ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(df.index, df['daily_return'], lw=0.6, color='steelblue', alpha=0.8)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_title('AAPL - Daily Returns (%)')
axes[0].set_ylabel('Return (%)')

axes[1].plot(df.index, df['vol_20'], label='20-day Volatility', color='red', lw=1)
axes[1].set_title('Rolling 20-day Volatility')
axes[1].set_ylabel('Std Dev (%)')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/fig8_returns_volatility.png', dpi=150)
plt.close()
print("Saved fig8_returns_volatility.png")


# ── 4. Multi-stock summary ────────────────────────────────────────────────────
print("\nBuilding multi-stock summary...")
summary = []
for t in TICKERS:
    try:
        d = load_stock(t)
        ret = d['Adj Close'].pct_change().dropna()
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
        total_ret = (d['Adj Close'].iloc[-1] / d['Adj Close'].iloc[0] - 1) * 100
        summary.append({
            'Ticker':              t,
            'Mean Daily Ret (%)':  round(ret.mean() * 100, 4),
            'Volatility (%)':      round(ret.std() * 100, 4),
            'Sharpe Ratio':        round(sharpe, 3),
            'Total Return (%)':    round(total_ret, 2),
        })
        print(f"  {t}: Sharpe={sharpe:.3f}, Total Return={total_ret:.1f}%")
    except Exception as e:
        print(f"  {t}: ERROR - {e}")

summary_df = pd.DataFrame(summary).set_index('Ticker')
print("\n=== Multi-Stock Summary ===")
print(summary_df.to_string())


# ── 5. Sharpe ratio bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

summary_df['Sharpe Ratio'].plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.8)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_title('Annualized Sharpe Ratio by Stock')
axes[0].set_ylabel('Sharpe Ratio')
axes[0].tick_params(axis='x', rotation=0)

summary_df['Total Return (%)'].plot(kind='bar', ax=axes[1], color='coral', alpha=0.8)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_title('Total Return (2020-2024) by Stock')
axes[1].set_ylabel('Return (%)')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{OUT}/fig9_multi_stock_summary.png', dpi=150)
plt.close()
print("\nSaved fig9_multi_stock_summary.png")

print("\nTask 2 complete. All figures saved to data/raw/")
