# Final Report — Predicting Price Moves with News Sentiment

**Submitted by:** [Your Name] | **Date:** May 2026 | **Branch:** `task-1`

---

## 1. Executive Summary

This report demonstrates how financial news headlines can be converted into actionable signals by combining sentiment analysis with technical indicators and price return correlation. Using 1.4 million financial news headlines between 2011 and 2020, the analysis shows that headline sentiment carries measurable information about same-day stock returns and that sentiment signals can be evaluated alongside traditional technical indicators like SMA, EMA, RSI, and MACD.

Key findings:

- The news corpus contains 1,407,328 headlines with strong publication concentration around key publishers and market hours.
- A pure pandas technical indicator pipeline successfully produced SMA(20/50), EMA(20), RSI(14), and MACD(12,26,9) for 7 major tickers.
- VADER sentiment correlates positively with same-day stock returns: overall Pearson r = 0.3319, p = 0.034.
- Oracle (ORCL) was the strongest stock-level signal with r = 0.90 (p = 0.006, n=7).
- Positive sentiment days averaged +1.56% return versus +0.11% on neutral days, indicating a directional effect.

This work provides Nova Financial Solutions with a reproducible analytical pipeline and evidence that news sentiment can be integrated into forecasting models.

---

## 2. Business Context

Nova Financial Solutions aims to enhance predictive analytics for investment teams by linking narrative market signals to price action. Financial headlines are published continuously, and the challenge is to separate signal from noise so that news sentiment can support trading decisions, risk management, and alpha generation.

The analysis focused on three core tasks:

1. Exploratory Data Analysis (EDA) on news headlines to identify structure, publisher bias, and topic distribution.
2. Technical indicator computation from historical stock price data to contextualize market momentum.
3. Correlation analysis between daily news sentiment and stock returns to quantify the predictive relationship.

---

## 3. Data and Preparation

### 3.1 News dataset

- Source: `raw_analyst_ratings.csv`
- Volume: 1,407,328 rows
- Fields: `headline`, `url`, `publisher`, `date`, `stock`
- Coverage: April 2011 to June 2020
- Data quality: zero missing values after date parsing and cleanup

### 3.2 Stock price dataset

- Stock tickers analyzed: AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA
- Source: historical CSV files downloaded from Yahoo Finance
- Frequency: daily trading days
- Fields: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

### 3.3 Data preparation steps

- Normalized dates and timestamps from the news dataset.
- Extracted publication hour and day-of-week features.
- Mapped legacy ticker `FB` to `META` for price matching.
- Aligned news trade dates to calendar trading days and removed unmatched rows.
- Computed daily returns using adjusted close prices.

---

## 4. Task 1 — Exploratory Data Analysis

### 4.1 Headline characteristics

- Average headline length: ~80 characters
- Average headline word count: ~12 words
- Distribution was right-skewed, with a minority of longer roundup-style headlines.

### 4.2 Publisher and timing insights

- Publisher coverage is concentrated, with the top publishers producing the largest article volumes.
- Articles are clustered during U.S. market hours, especially between 10:00 and 16:00 ET.
- Weekend publication is sparse, confirming that actionable market news is largely published on trading days.
- A significant article volume spike occurred in early 2020, consistent with COVID-19 market volatility.

### 4.3 Topic signals

- TF-IDF and bigram analysis revealed dominant themes like `price target`, `stocks hit`, earnings, and analyst rating actions.
- LDA topic modeling confirmed recurring themes around analyst ratings, earnings revisions, corporate guidance, and market sentiment.

### 4.4 Stock coverage

- The news dataset covers a broad set of tickers, with the top-covered names appearing frequently enough to support correlation analysis.
- The highest coverage tickers were those with active analyst attention and frequent news cycles.

---

## 5. Task 2 — Technical Indicators

### 5.1 Indicator approach

- SMA(20) and SMA(50) were computed to capture short- and medium-term trend structure.
- EMA(20) was computed for momentum-sensitive smoothing.
- RSI(14) was used to identify overbought/oversold conditions.
- MACD(12,26,9) was used to detect momentum shifts and crossover signals.
- All indicators were implemented in pure pandas, avoiding TA-Lib dependency issues on Windows.

### 5.2 Key technical findings

- AAPL exhibited a sustained bullish bias in the final year, with SMA(20) staying above SMA(50) and MACD mostly positive.
- RSI values for the analyzed tickers generally stayed within the 35–75 range, indicating balanced momentum rather than extreme overbought/oversold conditions.
- NVDA delivered the best risk-adjusted performance with an annualized Sharpe ratio of 1.25 and total return of 729% over the 2020–2024 period.
- TSLA delivered the highest raw return of 766%, but its daily volatility was the highest at 4.29%.

### 5.3 Visual analysis

- Figure 7 shows AAPL price alongside SMA(20), SMA(50), and EMA(20), with RSI and MACD panels below.
- Figure 8 shows AAPL daily returns and 20-day rolling volatility, highlighting the March 2020 market drawdown.
- Figure 9 summarizes Sharpe ratio and total return across the 7 selected tickers.

---

## 6. Task 3 — Correlation Analysis

### 6.1 Sentiment scoring

- VADER sentiment analysis was applied to each headline to generate a compound score in [-1, 1].
- Sentiment classes were assigned as Positive, Neutral, or Negative using standard VADER thresholds.
- Daily sentiment scores were averaged per stock and trading date to match daily returns.

### 6.2 Alignment and matching

- Price history was aligned with the news trade date, accounting for weekends and trading day coverage.
- Stocks were downloaded for the same date range as the news dataset to maximize overlap.
- After alignment, the merged dataset included matched daily sentiment and return observations for stocks with sufficient coverage.

### 6.3 Correlation results

- Overall correlation between average daily sentiment and same-day stock return: r = 0.3319, p = 0.034.
- This result is statistically significant at the 5% level and supports the hypothesis that news sentiment carries directional information.
- Oracle (ORCL) was the strongest single-stock signal with r = 0.90, p = 0.006.
- Positive sentiment days averaged +1.56% return, while neutral days averaged +0.11% return.

### 6.4 Lag analysis

- A lag analysis was performed for lag-0, lag-1, and lag-2 returns.
- The strongest same-day signal remains lag-0 overall, while some stocks show mean reversion patterns on lag-1.
- BABA exhibited a notable lag-1 reversal pattern (r = -0.81), suggesting that positive sentiment may be followed by pullback on the next day.

### 6.5 Visual analysis

- Figure 10 shows the sentiment vs. return scatter plot with the fitted OLS line and correlation annotation.
- Figure 11 shows per-stock Pearson correlation values for stocks with sufficient data.
- Figure 12 compares average returns for Positive, Neutral, and Negative sentiment days.
- Figure 13 compares lag-0, lag-1, and lag-2 correlations for stocks with at least 5 matched days.

---

## 7. Investment Strategy Recommendations

### 7.1 Sentiment-enhanced momentum

- Combine headline sentiment with technical momentum signals to improve entry timing.
- Use positive sentiment days as a bullish signal when technical indicators also show upward trend confirmation (e.g., SMA(20) above SMA(50), positive MACD).

### 7.2 Short-term sentiment overlay

- Use same-day sentiment strength to adjust position sizing or signal confidence.
- Positive sentiment days with high confidence may be suitable for short-term long exposures, while negative sentiment days can trigger defensive actions.

### 7.3 Stock selection guidance

- Prioritize stocks with stronger sentiment-return relationships, such as ORCL in this analysis.
- Treat weaker or noisy relationships as lower-conviction signals.

### 7.4 Risk controls

- Because TSLA showed high volatility, use tighter stop-loss rules or reduced leverage for high-volatility names.
- Consider limiting exposure on neutral sentiment days and focusing allocation on higher conviction setups.

---

## 8. Limitations

- The matched dataset remains limited by the overlap between news coverage and available price data.
- VADER is a general-purpose sentiment tool; finance-specific models such as FinBERT may capture nuance better.
- Correlation is not causation: headline sentiment may coincide with underlying news events rather than directly drive returns.
- The analysis uses same-day and short-lag returns; longer horizon effects and market microstructure factors are not covered.
- Sample size for some stocks is small, which can exaggerate stock-level correlation estimates.

---

## 9. Next Steps

1. Expand coverage to all available tickers in the news dataset to increase statistical power.
2. Compare VADER against finance-domain NLP models like FinBERT for improved sentiment precision.
3. Backtest combined sentiment and technical indicator strategies over a multi-year period.
4. Add event-level analysis around earnings, analyst revisions, and macro releases.
5. Build a production-ready pipeline for daily sentiment ingestion, signal scoring, and trading signal generation.

---

## 10. Repository and Reproducibility

- Code path: `src/`, `scripts/`, `notebooks/`
- Data path: `data/raw/`
- Tests: `tests/`, passing with `pytest`.
- CI: `.github/workflows/unittests.yml` runs unit tests on push and PR.
- Dependencies: `requirements.txt`.

The repository is ready for submission once the report and notebooks are verified, committed, and pushed to GitHub.
