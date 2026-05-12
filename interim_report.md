# Interim Report: Predicting Price Moves with News Sentiment

**Submitted by:** [Your Name] | **Date:** May 2026 | **Branch:** `task-1`

---

## 1. Executive Summary

This interim report presents a working pipeline that links financial news sentiment to same-day stock return behavior. Task 1 delivers a clean EDA on 1,407,328 headlines. Task 2 produces SMA(20/50), EMA(20), RSI(14), and MACD(12,26,9) for seven large-cap stocks using pure pandas. Task 3 demonstrates a statistically significant correlation between daily VADER sentiment and stock returns: overall Pearson r = 0.3319, p = 0.034.

---

## 2. Task 1 — Exploratory Data Analysis

**Dataset:** `raw_analyst_ratings.csv` — 1,407,328 rows, 5 columns, April 2011–June 2020.

- Headlines average 80 characters and 12 words, with a right-skewed length distribution driven by a small group of long roundup headlines.
- Publisher coverage is concentrated: the top sources account for the largest volume, creating a potential editorial bias.
- Publishing activity peaks during market hours and falls sharply on weekends.
- A large volume spike appears in early 2020, aligned with COVID-19 market events.
- Text analysis highlights financial themes such as `price target`, `stocks hit`, earnings, and analyst ratings.

![Figure 1: Headline length distributions (character count left, word count right).](data/raw/fig1_headline_distributions.png)

![Figure 2: Top 15 publishers by article count.](data/raw/fig2_top_publishers.png)

![Figure 3: Daily article volume 2011–2020, with the COVID-19 spike visible in early 2020.](data/raw/fig3_daily_volume.png)

![Figure 4: Articles by publication hour and day of week.](data/raw/fig4_publishing_times.png)

![Figure 5: Top 20 TF-IDF terms. Bigrams “price target” and “stocks hit” dominate.](data/raw/fig5_tfidf_terms.png)

![Figure 6: Top 20 most covered stock tickers.](data/raw/fig6_top_stocks.png)

---

## 3. Task 2 — Technical Indicators

**Data:** AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA, January 2020–January 2024.

- Technical indicators were implemented in native pandas for compatibility with Windows.
- SMA(20/50) and EMA(20) capture trend structure, RSI(14) measures momentum, and MACD(12,26,9) identifies crossover momentum shifts.
- AAPL’s final-year chart shows a sustained bullish structure: SMA(20) above SMA(50), moderate RSI, and largely positive MACD.
- NVDA produced the strongest risk-adjusted return; TSLA delivered the highest raw return with greater volatility.

![Figure 7: AAPL technical indicator panel with price, moving averages, RSI, and MACD.](data/raw/fig7_technical_indicators.png)

![Figure 8: AAPL daily returns and 20-day rolling volatility, full 2020–2024 period.](data/raw/fig8_returns_volatility.png)

![Figure 9: Annualized Sharpe ratio and total return for the seven stocks.](data/raw/fig9_multi_stock_summary.png)

---

## 4. Task 3 — Correlation Analysis

**Method:** VADER sentiment scores were averaged by stock and trade date, then merged with same-day returns.

- Price history and news dates were aligned carefully; legacy `FB` ticker values were mapped to `META`.
- Overall sentiment-return correlation is positive and statistically significant: r = 0.3319, p = 0.034.
- ORCL stands out with a strong single-stock signal: r = 0.90, p = 0.006.
- Positive-sentiment days generated meaningfully higher returns than neutral days.
- Lag analysis suggests some stocks can exhibit mean-reversion on the next day, notably BABA.

![Figure 10: Sentiment vs. same-day return scatter plot with fitted trend line.](data/raw/fig10_sentiment_vs_return.png)

![Figure 11: Per-stock Pearson r for same-day sentiment vs. return.](data/raw/fig11_correlation_by_stock.png)

![Figure 12: Average daily return by sentiment category.](data/raw/fig12_return_by_sentiment_class.png)

![Figure 13: Same-day versus lag-1 and lag-2 correlation by stock.](data/raw/fig13_lag_correlation.png)

---

## 5. Challenges

| Challenge                        | Resolution                             |
| -------------------------------- | -------------------------------------- |
| News/stock date mismatch         | Re-downloaded stock data for 2011–2020 |
| Legacy ticker mapping            | Mapped `FB` to `META`                  |
| TA-Lib dependency on Windows     | Implemented indicators in pandas       |
| Sparse coverage for some tickers | Focused on top-covered stocks          |

---

## 6. Summary and Next Steps

- Task 1 is complete with robust EDA and aligned visual evidence.
- Task 2 includes the required technical indicator visualization (Figure 7) and a cross-stock performance summary.
- Task 3 provides a clear sentiment-return correlation analysis with both overall and per-stock results.

Next steps:

1. Expand matching to additional tickers for stronger statistical power.
2. Test finance-specific sentiment models such as FinBERT.
3. Backtest combined sentiment and technical signals.

---

_This report is intentionally concise and aligned to the figures generated in `data/raw/` for submission readiness._
