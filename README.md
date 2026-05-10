# News Sentiment Analysis — Predicting Price Moves with News Sentiment

A rigorous analytical pipeline that quantifies sentiment in financial news headlines, computes technical indicators from historical stock price data, and measures the statistical relationship between the two.

## Project Structure

```
news-sentiment-analysis/
├── .github/workflows/unittests.yml   # CI/CD pipeline
├── data/raw/                         # Raw datasets (not committed)
├── notebooks/                        # Jupyter analysis notebooks
├── src/                              # Reusable Python modules
├── scripts/                          # Standalone scripts
└── tests/                            # Unit tests
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Tasks

| Task | Description |
|------|-------------|
| Task 1 | EDA — descriptive stats, topic modeling, time-series publication analysis |
| Task 2 | Technical indicators — SMA, EMA, RSI, MACD using TA-Lib & PyNance |
| Task 3 | Correlation — news sentiment vs. daily stock returns (Pearson) |

## Data

Place raw data files under `data/raw/`:
- `raw_analyst_ratings.csv` — financial news headlines dataset
- `AAPL_historical.csv`, `AMZN_historical.csv`, etc. — stock price CSVs

## Running Tests

```bash
pytest tests/
```
