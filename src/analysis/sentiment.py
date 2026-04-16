"""
Sentiment Analysis Module

Document and sentence-level sentiment using NLTK VADER.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

_COMPOUND_POSITIVE = 0.05
_COMPOUND_NEGATIVE = -0.05


def _ensure_nltk_sentiment_resources() -> None:
    """Download VADER lexicon and sentence tokenizer data if missing."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        logger.info("Downloading NLTK vader_lexicon...")
        nltk.download("vader_lexicon", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _label_from_compound(compound: float) -> str:
    if compound >= _COMPOUND_POSITIVE:
        return "positive"
    if compound <= _COMPOUND_NEGATIVE:
        return "negative"
    return "neutral"


@dataclass
class SentimentResult:
    """VADER sentiment scores for full text and per sentence."""

    compound: float
    positive: float
    negative: float
    neutral: float
    label: str
    sentence_sentiments: list[dict[str, str | float]] = field(default_factory=list)
    distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall": {
                "compound": self.compound,
                "positive": self.positive,
                "negative": self.negative,
                "neutral": self.neutral,
                "label": self.label,
            },
            "sentence_sentiments": self.sentence_sentiments,
            "distribution": self.distribution,
        }


class SentimentAnalyzer:
    """
    Sentiment analysis using NLTK VADER.

    Computes polarity on the full extracted text and per sentence.
    """

    def __init__(self) -> None:
        _ensure_nltk_sentiment_resources()
        self._sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of full text and each sentence.

        Args:
            text: Raw document text (e.g. extracted page content).

        Returns:
            SentimentResult with overall scores, per-sentence rows, and label counts.
        """
        stripped = (text or "").strip()
        if not stripped:
            return SentimentResult(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                label="neutral",
                sentence_sentiments=[],
                distribution={"positive": 0, "negative": 0, "neutral": 0},
            )

        overall = self._sia.polarity_scores(stripped)
        overall_label = _label_from_compound(overall["compound"])

        try:
            sentences = sent_tokenize(stripped)
        except Exception as exc:
            logger.warning("Sentence tokenization failed: %s", exc)
            sentences = [stripped]
        if not sentences:
            sentences = [stripped]

        sentence_rows: list[dict[str, str | float]] = []
        dist = {"positive": 0, "negative": 0, "neutral": 0}

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            scores = self._sia.polarity_scores(s)
            lbl = _label_from_compound(scores["compound"])
            dist[lbl] = dist.get(lbl, 0) + 1
            sentence_rows.append({
                "sentence": s[:500] + ("..." if len(s) > 500 else ""),
                "compound": scores["compound"],
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "label": lbl,
            })

        return SentimentResult(
            compound=overall["compound"],
            positive=overall["pos"],
            negative=overall["neg"],
            neutral=overall["neu"],
            label=overall_label,
            sentence_sentiments=sentence_rows,
            distribution=dist,
        )
