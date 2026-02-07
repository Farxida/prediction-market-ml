"""Sentiment analysis pipeline for Polymarket NLP features.

Models:
- ProsusAI/finbert: financial domain, trained on 10K/10Q SEC filings
- distilbert-base-uncased-finetuned-sst-2-english: general sentiment (fallback)

Design decisions:
- AI-Trader paper: LLM should not make buy/sell decisions -- use sentiment as FEATURE
- FinBERT > general BERT for financial text (domain-specific pre-training)
- Batch processing for efficiency (not real-time per comment)
- MPS (Apple Silicon GPU) acceleration

Output: sentiment scores (-1 to +1) for each text, plus aggregated features.
"""

import os
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# Fix torch MPS duplicate lib issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class SentimentResult:
    """Single text sentiment result."""
    text: str
    label: str           # "positive", "negative", "neutral"
    score: float         # confidence 0-1
    sentiment: float     # -1 to +1 (negative to positive)


class SentimentAnalyzer:
    """FinBERT-based sentiment analysis with MPS acceleration.

    Uses ProsusAI/finbert (financial domain) as primary model.
    Falls back to distilbert-sst2 if finbert unavailable.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None
        self._device = device

    def _load(self):
        """Lazy-load the model (avoid loading at import time)."""
        if self._pipeline is not None:
            return

        from transformers import pipeline
        import torch

        if self._device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self._device

        try:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
                truncation=True,
                max_length=512,
            )
            log.info(f"Loaded {self.model_name} on {device}")
        except Exception as e:
            log.warning(f"Failed to load {self.model_name}: {e}, falling back to distilbert")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device,
                truncation=True,
                max_length=512,
            )
            self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            log.info(f"Loaded fallback model on {device}")

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        self._load()
        result = self._pipeline(text)[0]
        return self._to_result(text, result)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts efficiently.

        Processes in batches for GPU efficiency.
        """
        self._load()
        if not texts:
            return []

        # Filter empty/None texts
        valid = [(i, t) for i, t in enumerate(texts) if t and len(t.strip()) > 0]
        if not valid:
            return []

        indices, clean_texts = zip(*valid)
        results = self._pipeline(
            list(clean_texts),
            batch_size=self.batch_size,
            truncation=True,
            max_length=512,
        )

        return [self._to_result(t, r) for t, r in zip(clean_texts, results)]

    def _to_result(self, text: str, raw: dict) -> SentimentResult:
        """Convert pipeline output to SentimentResult."""
        label = raw["label"].lower()
        score = raw["score"]

        # Map to -1 to +1 scale
        if label in ("positive", "pos"):
            sentiment = score
        elif label in ("negative", "neg"):
            sentiment = -score
        else:  # neutral
            sentiment = 0.0

        return SentimentResult(
            text=text[:200],  # truncate for storage
            label=label,
            score=score,
            sentiment=sentiment,
        )

    def aggregate(self, results: list[SentimentResult]) -> dict:
        """Aggregate sentiment results into features.

        Returns dict of NLP features for a single market/event.
        """
        if not results:
            return {
                "nlp_sentiment_mean": 0.0,
                "nlp_sentiment_std": 0.0,
                "nlp_positive_ratio": 0.0,
                "nlp_negative_ratio": 0.0,
                "nlp_neutral_ratio": 0.0,
                "nlp_text_count": 0,
                "nlp_sentiment_strength": 0.0,
            }

        sentiments = [r.sentiment for r in results]
        labels = [r.label for r in results]
        n = len(results)

        return {
            "nlp_sentiment_mean": float(np.mean(sentiments)),
            "nlp_sentiment_std": float(np.std(sentiments)) if n > 1 else 0.0,
            "nlp_positive_ratio": labels.count("positive") / n,
            "nlp_negative_ratio": labels.count("negative") / n,
            "nlp_neutral_ratio": labels.count("neutral") / n,
            "nlp_text_count": n,
            "nlp_sentiment_strength": float(np.mean(np.abs(sentiments))),
        }
