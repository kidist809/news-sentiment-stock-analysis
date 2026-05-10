"""
Task 1 — Exploratory Data Analysis
Run: venv/Scripts/python scripts/run_task1_eda.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

plt.style.use('seaborn-v0_8-whitegrid')
STOP_WORDS = set(stopwords.words('english'))
OUT = 'data/raw'

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(f'{OUT}/newsData/raw_analyst_ratings.csv', index_col=0)
print(f"Shape: {df.shape}")

df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
df.dropna(subset=['date'], inplace=True)
df['date_only']  = df['date'].dt.date
df['hour']       = df['date'].dt.hour
df['year']       = df['date'].dt.year
df['dayofweek']  = df['date'].dt.day_name()
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# ── 2. Descriptive Statistics ──────────────────────────────────────────────────
print("=== Headline Character Count ===")
df['headline_len'] = df['headline'].str.len()
df['word_count']   = df['headline'].str.split().str.len()
print(df['headline_len'].describe().round(2))
print("\n=== Word Count ===")
print(df['word_count'].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['headline_len'].dropna(), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Headline Character Length Distribution')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Count')
axes[1].hist(df['word_count'].dropna(), bins=30, color='coral', edgecolor='white')
axes[1].set_title('Headline Word Count Distribution')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_headline_distributions.png', dpi=150)
plt.close()
print("Saved fig1_headline_distributions.png")

# ── 3. Publisher Analysis ──────────────────────────────────────────────────────
def extract_domain(pub):
    if isinstance(pub, str) and '@' in pub:
        return pub.split('@')[-1].lower()
    return pub

df['publisher_clean'] = df['publisher'].apply(extract_domain)
top_publishers = df['publisher_clean'].value_counts().head(15)
print(f"\n=== Top 15 Publishers ===\n{top_publishers}")

fig, ax = plt.subplots(figsize=(10, 5))
top_publishers.plot(kind='barh', ax=ax, color='teal')
ax.set_title('Top 15 Most Active Publishers')
ax.set_xlabel('Article Count')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_top_publishers.png', dpi=150)
plt.close()
print("Saved fig2_top_publishers.png")

# ── 4. Time Series — Daily Volume ─────────────────────────────────────────────
daily_counts = df.groupby('date_only').size().reset_index(name='count')
daily_counts['date_only'] = pd.to_datetime(daily_counts['date_only'])

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily_counts['date_only'], daily_counts['count'], linewidth=0.8, color='navy', alpha=0.7)
ax.fill_between(daily_counts['date_only'], daily_counts['count'], alpha=0.2, color='navy')
ax.set_title('Daily Article Publication Volume Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Articles Published')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_daily_volume.png', dpi=150)
plt.close()
print("Saved fig3_daily_volume.png")

# Publishing hour + day of week
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['hour'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Articles by Hour of Day (UTC-4)')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Count')

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dayofweek'].value_counts().reindex(day_order).plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Articles by Day of Week')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_publishing_times.png', dpi=150)
plt.close()
print("Saved fig4_publishing_times.png")

# ── 5. Text Analysis — TF-IDF ─────────────────────────────────────────────────
print("\nRunning TF-IDF (sample of 50k rows for speed)...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

sample = df['headline'].dropna().sample(min(50000, len(df)), random_state=42)
cleaned = sample.apply(clean_text)

tfidf = TfidfVectorizer(max_features=30, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(cleaned)
tfidf_scores = tfidf_matrix.mean(axis=0).A1
tfidf_terms = pd.Series(tfidf_scores, index=tfidf.get_feature_names_out()).sort_values(ascending=False)

print(f"\n=== Top 20 TF-IDF Terms ===\n{tfidf_terms.head(20)}")

fig, ax = plt.subplots(figsize=(10, 5))
tfidf_terms.head(20).plot(kind='barh', ax=ax, color='mediumseagreen')
ax.set_title('Top 20 TF-IDF Terms in Headlines')
ax.set_xlabel('Mean TF-IDF Score')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_tfidf_terms.png', dpi=150)
plt.close()
print("Saved fig5_tfidf_terms.png")

# ── 6. LDA Topic Modeling ──────────────────────────────────────────────────────
print("\nRunning LDA topic modeling...")
cv = CountVectorizer(max_features=500, ngram_range=(1, 2))
cv_matrix = cv.fit_transform(cleaned)
lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10)
lda.fit(cv_matrix)
feature_names = cv.get_feature_names_out()
print("\n=== LDA Topics ===")
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]
    print(f"Topic {i+1}: {', '.join(top_words)}")

# ── 7. Stock Coverage ──────────────────────────────────────────────────────────
top_stocks = df['stock'].value_counts().head(20)
print(f"\nUnique stocks covered: {df['stock'].nunique()}")
print(f"Top 10 stocks:\n{top_stocks.head(10)}")

fig, ax = plt.subplots(figsize=(10, 5))
top_stocks.plot(kind='bar', ax=ax, color='slateblue')
ax.set_title('Top 20 Most Covered Stocks')
ax.set_xlabel('Stock Ticker')
ax.set_ylabel('Article Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_top_stocks.png', dpi=150)
plt.close()
print("Saved fig6_top_stocks.png")

print("\nTask 1 EDA complete. All figures saved to data/raw/")
