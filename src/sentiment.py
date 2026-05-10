"""
Sentiment analysis utilities using VADER.
"""
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()


def score_headline(headline: str) -> float:
    """Return VADER compound score [-1, 1] for a single headline."""
    return _sia.polarity_scores(str(headline))['compound']


def classify_sentiment(score: float) -> str:
    """Classify a compound score as Positive, Neutral, or Negative."""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'


def score_dataframe(df, headline_col: str = 'headline') -> 'pd.DataFrame':
    """
    Add 'sentiment' and 'sentiment_class' columns to a DataFrame.
    Returns a copy of the DataFrame.
    """
    import pandas as pd
    out = df.copy()
    out['sentiment']       = out[headline_col].apply(score_headline)
    out['sentiment_class'] = out['sentiment'].apply(classify_sentiment)
    return out
