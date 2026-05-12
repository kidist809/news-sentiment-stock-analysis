# Interim Report: Predicting Price Moves with News Sentiment
**Submitted by:** [Your Name] | **Date:** May 2026 | **Branch:** `task-1`

---

## 1. Executive Summary

All three tasks are complete. Task 1 performed EDA on 1,407,328 financial news headlines (2011–2020). Task 2 computed SMA(20/50), EMA(20), RSI(14), and MACD(12,26,9) for 7 stocks (2020–2024). Task 3 found a statistically significant Pearson correlation of r = 0.33 (p = 0.034) between daily VADER sentiment and same-day stock returns, with Oracle showing r = 0.90 (p = 0.006).

---

## 2. Task 1 — Exploratory Data Analysis

**Dataset:** `raw_analyst_ratings.csv` — 1,407,328 rows, 5 columns, April 2011–June 2020, zero missing values.

Headlines average 80 characters and 12 words (median 63 chars, 10 words). The right-skewed distribution reflects a minority of long multi-company roundup articles. Benzinga dominates publishing — the top 2 sources alone account for 27,158 articles, introducing an editorial bias to note in any NLP model. Publication volume peaks during US market hours (10am–4pm ET) and drops sharply on weekends. A sharp spike in early 2020 aligns with COVID-19 market volatility. TF-IDF bigram analysis identified `price target` and `stocks hit` as the dominant news categories; LDA (5 topics) confirmed analyst ratings and earnings reports as the most actionable topics for sentiment-price correlation.

![Figure 1: Headline length distributions (character count left, word count right).](data/raw/fig1_headline_distributions.png)

![Figure 2: Top 15 publishers by article count.](data/raw/fig2_top_publishers.png)

![Figure 3: Daily article volume 2011–2020. COVID-19 spike visible in early 2020.](data/raw/fig3_daily_volume.png)

![Figure 4: Articles by hour of day (left) and day of week (right).](data/raw/fig4_publishing_times.png)

![Figure 5: Top 20 TF-IDF terms. Bigrams "price target" and "stocks hit" dominate.](data/raw/fig5_tfidf_terms.png)

![Figure 6: Top 20 most covered stock tickers.](data/raw/fig6_top_stocks.png)

---

## 3. Task 2 — Technical Indicators

**Data:** 7 tickers (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA), Jan 2020–Jan 2024, 1,006 trading days each. All indicators implemented in pure pandas (TA-Lib unavailable on Windows).

Figure 7 shows AAPL over the final year: SMA(20) remained above SMA(50) — a sustained bullish crossover; RSI stayed between 35–75 (no extremes); MACD was mostly positive in H2, confirming upward momentum. Figure 8 shows the full 2020–2024 daily returns and 20-day rolling volatility — the COVID-19 crash (March 2020) is visible as a cluster of single-day returns below −10%. NVDA delivered the best risk-adjusted return (Sharpe 1.25, total return 729%); TSLA had the highest raw return (766%) but also the highest volatility (4.29% daily std dev).

![Figure 7: AAPL — Close price with SMA(20/50) and EMA(20) (top panel); RSI(14) with 70/30 bands (middle); MACD(12,26,9) line, signal, and histogram (bottom). Final 365 days of dataset.](data/raw/fig7_technical_indicators.png)

![Figure 8: AAPL daily returns % (top) and 20-day rolling volatility (bottom), full 2020–2024 period.](data/raw/fig8_returns_volatility.png)

![Figure 9: Annualized Sharpe ratio (left) and total return % (right) for all 7 tickers, 2020–2024.](data/raw/fig9_multi_stock_summary.png)

---

## 4. Task 3 — Correlation Analysis

**Date alignment:** The news dataset (2011–2020) and initial stock data (2020–2024) overlapped by only 38 days. Stock data was re-downloaded for 2011–2020. Meta's pre-2021 ticker `FB` in the news dataset was mapped to `META` for price downloads. After merging daily average VADER sentiment with daily returns, 41 matched trading days were available across 7 stocks.

**Overall: r = 0.3319, p = 0.034** — statistically significant at the 5% level. ORCL is the standout with r = 0.90 (p = 0.006, n = 7). FB's apparent r = −0.97 is unreliable (n = 3 only). Positive-sentiment days averaged +1.56% return vs. +0.11% on neutral days — a 14× difference. Lag-1 analysis revealed a mean-reversion pattern for BABA (lag-1 r = −0.81): positive news on day 0 is followed by negative returns on day 1.

![Figure 10: Scatter plot — daily VADER sentiment vs. daily return (%). OLS fit: r = 0.332, p = 0.034.](data/raw/fig10_sentiment_vs_return.png)

![Figure 11: Per-stock Pearson r (same-day). Green = positive, red = negative correlation.](data/raw/fig11_correlation_by_stock.png)

![Figure 12: Average daily return by sentiment category (Positive / Neutral / Negative).](data/raw/fig12_return_by_sentiment_class.png)

![Figure 13: Same-day (blue) vs. lag-1 (coral) Pearson r for stocks with ≥ 5 matched days.](data/raw/fig13_lag_correlation.png)

---

## 5. Challenges

| Challenge | Resolution |
|-----------|-----------|
| News/stock date range mismatch (38-day overlap) | Re-downloaded stock data for 2011–2020 |
| `FB` ticker in news vs. `META` in price data | Ticker mapping dictionary |
| TA-Lib C binary unavailable on Windows | Pure pandas reimplementation |
| Sparse news coverage for initial 7 tickers | Expanded to top 10 tickers by news volume |
| `DataFrame.last()` deprecated in pandas 3.0 | Replaced with boolean index slicing |

---

## 6. Next Steps

1. **Expand dataset:** Download price data for all 6,204 news tickers to increase matched pairs from ~41 to thousands for robust per-stock correlations.
2. **Benchmark sentiment tools:** Compare VADER against FinBERT to assess whether finance-domain embeddings improve correlation strength.
3. **Multi-lag analysis:** Test lag-0 through lag-3 across all stocks to identify the optimal prediction horizon.
4. **Composite signal:** Combine RSI/MACD crossover signals with sentiment scores and backtest against a buy-and-hold baseline.
5. **Final report:** Consolidate findings into a publication-quality report with actionable trading rules based on the ORCL (r = 0.90) and BABA mean-reversion findings.
