"""NLP feature engineering for Polymarket markets.

Combines comment sentiment + news sentiment + text statistics
into features for the ML pipeline.

Research basis:
- Comment volume and sentiment change are most predictive
- Pre-event sentiment > post-event
- Whale/expert comments > crowd (weight by author reputation)
- Combine with price features in ensemble (not standalone)

Features produced (per market):
  nlp_comment_count         — number of comments
  nlp_comment_sentiment     — mean comment sentiment (-1 to +1)
  nlp_comment_sentiment_std — sentiment disagreement
  nlp_comment_positive_ratio — fraction positive
  nlp_comment_velocity      — comments per hour (recent)
  nlp_news_count            — number of news articles
  nlp_news_sentiment        — mean news sentiment
  nlp_sentiment_divergence  — news vs comment sentiment gap
  nlp_mention_frequency     — relative mention volume
  nlp_bullish_keyword_ratio — "yes"/"will"/"bullish" ratio
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.features.sentiment import SentimentAnalyzer, SentimentResult
from src.utils.logger import get_logger

log = get_logger(__name__)

# Bullish/bearish keywords for prediction markets
BULLISH_KEYWORDS = {
    "yes", "will", "bullish", "likely", "definitely", "certain",
    "confirmed", "guaranteed", "inevitable", "obvious", "100%",
    "easy", "lock", "sure", "slam dunk",
}
BEARISH_KEYWORDS = {
    "no", "won't", "bearish", "unlikely", "never", "impossible",
    "doubt", "overpriced", "bubble", "scam", "0%", "waste",
    "no chance", "not going",
}


def extract_comment_features(
    comments: list[dict],
    sentiment_results: list[SentimentResult] | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, float]:
    """Extract NLP features from a list of comments.

    Args:
        comments: list of comment dicts (from Comments API)
        sentiment_results: pre-computed sentiment (optional)
        analyzer: SentimentAnalyzer instance (if sentiment_results not provided)
    """
    if not comments:
        return _empty_comment_features()

    # Extract text from comments (API returns "body" field)
    texts = []
    for c in comments:
        text = c.get("body", "") or c.get("content", "") or c.get("text", "")
        if text and len(text.strip()) > 2:  # skip very short
            texts.append(text)

    if not texts:
        return _empty_comment_features()

    # Get sentiment
    if sentiment_results is None and analyzer is not None:
        sentiment_results = analyzer.analyze_batch(texts)
    elif sentiment_results is None:
        sentiment_results = []

    # Keyword analysis
    all_text = " ".join(texts).lower()
    word_count = max(len(all_text.split()), 1)
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in all_text)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in all_text)
    keyword_total = max(bullish_count + bearish_count, 1)

    # Comment velocity (comments per hour)
    velocity = _compute_velocity(comments)

    # Sentiment features
    if sentiment_results:
        sentiments = [r.sentiment for r in sentiment_results]
        features = {
            "nlp_comment_count": len(texts),
            "nlp_comment_sentiment": float(np.mean(sentiments)),
            "nlp_comment_sentiment_std": float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
            "nlp_comment_positive_ratio": sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            "nlp_comment_velocity": velocity,
            "nlp_bullish_keyword_ratio": bullish_count / keyword_total,
        }
    else:
        features = {
            "nlp_comment_count": len(texts),
            "nlp_comment_sentiment": 0.0,
            "nlp_comment_sentiment_std": 0.0,
            "nlp_comment_positive_ratio": 0.0,
            "nlp_comment_velocity": velocity,
            "nlp_bullish_keyword_ratio": bullish_count / keyword_total,
        }

    return features


def extract_news_features(
    articles: list[dict],
    sentiment_results: list[SentimentResult] | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, float]:
    """Extract NLP features from news articles."""
    if not articles:
        return _empty_news_features()

    texts = [a.get("title", "") for a in articles if a.get("title")]
    if not texts:
        return _empty_news_features()

    if sentiment_results is None and analyzer is not None:
        sentiment_results = analyzer.analyze_batch(texts)

    if sentiment_results:
        sentiments = [r.sentiment for r in sentiment_results]
        return {
            "nlp_news_count": len(texts),
            "nlp_news_sentiment": float(np.mean(sentiments)),
        }

    return {
        "nlp_news_count": len(texts),
        "nlp_news_sentiment": 0.0,
    }


def combine_nlp_features(
    comment_features: dict[str, float],
    news_features: dict[str, float],
) -> dict[str, float]:
    """Combine comment and news features, add cross-source features."""
    features = {}
    features.update(comment_features)
    features.update(news_features)

    # Cross-source: sentiment divergence (news vs comments)
    cs = comment_features.get("nlp_comment_sentiment", 0.0)
    ns = news_features.get("nlp_news_sentiment", 0.0)
    features["nlp_sentiment_divergence"] = abs(cs - ns)

    # Relative mention volume
    comment_n = comment_features.get("nlp_comment_count", 0)
    news_n = news_features.get("nlp_news_count", 0)
    total = max(comment_n + news_n, 1)
    features["nlp_mention_frequency"] = total

    return features


def _compute_velocity(comments: list[dict]) -> float:
    """Compute comments per hour from timestamps."""
    timestamps = []
    for c in comments:
        ts = c.get("createdAt") or c.get("created_at") or c.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts)
                else:
                    continue
                timestamps.append(dt)
            except (ValueError, OSError):
                continue

    if len(timestamps) < 2:
        return 0.0

    time_span = (max(timestamps) - min(timestamps)).total_seconds()
    if time_span < 60:  # less than a minute
        return 0.0
    hours = time_span / 3600
    return len(timestamps) / hours


def _empty_comment_features() -> dict[str, float]:
    return {
        "nlp_comment_count": 0,
        "nlp_comment_sentiment": 0.0,
        "nlp_comment_sentiment_std": 0.0,
        "nlp_comment_positive_ratio": 0.0,
        "nlp_comment_velocity": 0.0,
        "nlp_bullish_keyword_ratio": 0.0,
    }


def _empty_news_features() -> dict[str, float]:
    return {
        "nlp_news_count": 0,
        "nlp_news_sentiment": 0.0,
    }
