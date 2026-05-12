"""
Task 2 — Quantitative Analysis with Technical Indicators & PyNance-style Metrics
Run: venv/Scripts/python scripts/run_task2_indicators.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from indicators import add_indicators, pynance_metrics

plt.style.use('seaborn-v0_8-whitegrid')

TICKERS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
OUT = 'data/raw'


def load_stock(ticker):
    df = pd.read_csv(f'{OUT}/{ticker}_historical.csv', header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df = df[~df.index.isin(['Ticker', 'Date', ''])]
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Close', 'Adj Close'], inplace=True)
    df.sort_index(inplace=True)
    return df


# ── 1. Compute indicators and PyNance metrics for all tickers ─────────────────
print("Computing indicators for all tickers...\n")
summary = []

for ticker in TICKERS:
    print(f"Processing {ticker}...")
    df = load_stock(ticker)
    df = add_indicators(df)
    metrics = pynance_metrics(df)
    metrics['Ticker'] = ticker
    summary.append(metrics)
    print(f"  Sharpe={metrics['sharpe_ratio']}  Total Return={metrics['total_return_pct']}%")

    # ── Per-ticker: Price + RSI + MACD (last 365 days) ────────────────────────
    plot_df = df[df.index >= df.index.max() - pd.DateOffset(days=365)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    axes[0].plot(plot_df.index, plot_df['Close'],      label='Close',  lw=1.2, color='black')
    axes[0].plot(plot_df.index, plot_df['SMA_20'],     label='SMA 20', lw=1,   color='blue',   ls='--')
    axes[0].plot(plot_df.index, plot_df['SMA_50'],     label='SMA 50', lw=1,   color='orange', ls='--')
    axes[0].plot(plot_df.index, plot_df['EMA_20'],     label='EMA 20', lw=1,   color='green',  ls=':')
    axes[0].set_title(f'{ticker} — Price & Moving Averages (SMA 20/50, EMA 20)')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=9)

    axes[1].plot(plot_df.index, plot_df['RSI_14'], color='purple', lw=1)
    axes[1].axhline(70, color='red',   ls='--', lw=0.8, label='Overbought (70)')
    axes[1].axhline(30, color='green', ls='--', lw=0.8, label='Oversold (30)')
    axes[1].set_ylabel('RSI (14)')
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)

    axes[2].plot(plot_df.index, plot_df['MACD'],        label='MACD',   color='blue',   lw=1)
    axes[2].plot(plot_df.index, plot_df['MACD_signal'], label='Signal', color='orange', lw=1)
    axes[2].bar(plot_df.index,  plot_df['MACD_hist'],   label='Hist',   color='gray',   alpha=0.4, width=1)
    axes[2].axhline(0, color='black', lw=0.5)
    axes[2].set_ylabel('MACD (12,26,9)')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fname = f'{OUT}/fig_indicators_{ticker.lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

# ── 2. AAPL returns & volatility (full period) ────────────────────────────────
print("\nGenerating AAPL returns/volatility chart...")
df_aapl = load_stock('AAPL')
df_aapl = add_indicators(df_aapl)
pynance_metrics(df_aapl)   # adds daily_return, vol_20, vol_50

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(df_aapl.index, df_aapl['daily_return'], lw=0.6, color='steelblue', alpha=0.8)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_title('AAPL — Daily Returns (%) 2020–2024')
axes[0].set_ylabel('Return (%)')

axes[1].plot(df_aapl.index, df_aapl['vol_20'], label='20-day Vol', color='red',    lw=1)
axes[1].plot(df_aapl.index, df_aapl['vol_50'], label='50-day Vol', color='orange', lw=1)
axes[1].set_title('Rolling Volatility')
axes[1].set_ylabel('Std Dev (%)')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/fig8_returns_volatility.png', dpi=150)
plt.close()
print("Saved fig8_returns_volatility.png")

# ── 3. Multi-stock summary chart ──────────────────────────────────────────────
summary_df = pd.DataFrame(summary).set_index('Ticker')
print("\n=== Multi-Stock Summary ===")
print(summary_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
summary_df['sharpe_ratio'].plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.8)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_title('Annualized Sharpe Ratio by Stock (2020–2024)')
axes[0].set_ylabel('Sharpe Ratio')
axes[0].tick_params(axis='x', rotation=0)

summary_df['total_return_pct'].plot(kind='bar', ax=axes[1], color='coral', alpha=0.8)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_title('Total Return (%) by Stock (2020–2024)')
axes[1].set_ylabel('Return (%)')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{OUT}/fig9_multi_stock_summary.png', dpi=150)
plt.close()
print("Saved fig9_multi_stock_summary.png")

# Save fig7 as AAPL (the primary indicator chart for the report)
os.replace(f'{OUT}/fig_indicators_aapl.png', f'{OUT}/fig7_technical_indicators.png')
print("Saved fig7_technical_indicators.png (AAPL)")

print("\nTask 2 complete.")
