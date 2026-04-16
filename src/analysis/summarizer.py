"""
Text summarization: OpenAI abstractive summary with TF-IDF extractive fallback.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_OPENAI_CHARS = 14_000
_EXTRACTIVE_MIN = 3
_EXTRACTIVE_MAX = 5
_OPENAI_MODEL = "gpt-3.5-turbo"

_CITATION_BRACKET = re.compile(r"\[\d+\]")
_MIN_SENTENCE_CHARS = 20
_MAX_SEPARATORS_FOR_PROSE = 5


def _strip_citation_brackets(text: str) -> str:
    """Remove Wikipedia-style numeric citations like [83] from text."""
    cleaned = _CITATION_BRACKET.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_extractive_candidate(sentence: str) -> bool:
    """Drop citation-heavy, list-like, or too-short sentences before TF-IDF."""
    s = sentence.strip()
    if len(s) < _MIN_SENTENCE_CHARS:
        return False
    if _CITATION_BRACKET.search(s):
        return False
    if s.count(",") > _MAX_SEPARATORS_FOR_PROSE:
        return False
    if s.count(";") > _MAX_SEPARATORS_FOR_PROSE:
        return False
    return True


def _filter_extractive_sentences(sentences: list[str]) -> list[str]:
    """Keep sentences suitable for extractive summarization."""
    filtered = [s.strip() for s in sentences if _is_extractive_candidate(s)]
    if filtered:
        return filtered
    # Degenerate: keep longer fragments without citation brackets
    fallback = [
        s.strip()
        for s in sentences
        if len(s.strip()) >= _MIN_SENTENCE_CHARS and not _CITATION_BRACKET.search(s)
    ]
    return fallback if fallback else [s.strip() for s in sentences if s.strip()]


@dataclass
class SummaryResult:
    """Summarization output."""

    summary: str
    method: str
    sentence_count: int

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "method": self.method,
            "sentence_count": self.sentence_count,
        }


class TextSummarizer:
    """
    Produces a short summary using OpenAI when configured, else TF-IDF extractive.
    """

    def __init__(self, openai_model: str = _OPENAI_MODEL) -> None:
        self._openai_model = openai_model

    def _openai_key_ok(self) -> bool:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        return bool(key)

    def _summarize_openai(self, text: str) -> str | None:
        if not self._openai_key_ok():
            return None
        if OpenAI is None:
            logger.warning("openai package not available")
            return None

        payload = text.strip()
        if len(payload) > _MAX_OPENAI_CHARS:
            payload = payload[:_MAX_OPENAI_CHARS]

        system = (
            "You summarize web article text for readers. Respond with a concise "
            "summary of exactly 3 to 5 complete sentences. No bullet points, "
            "no preamble, no markdown—plain prose only."
        )
        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": payload},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            choice = completion.choices[0].message.content
            if not choice:
                return None
            return choice.strip()
        except Exception as exc:
            logger.warning("OpenAI summarization failed: %s", exc)
            return None

    def _summarize_extractive(self, text: str) -> tuple[str, int]:
        stripped = (text or "").strip()
        if not stripped:
            return "", 0

        try:
            raw_sentences = [s.strip() for s in sent_tokenize(stripped) if s.strip()]
        except Exception as exc:
            logger.warning("sent_tokenize failed in summarizer: %s", exc)
            raw_sentences = [stripped]

        if not raw_sentences:
            return "", 0

        sentences = _filter_extractive_sentences(raw_sentences)

        if not sentences:
            return "", 0

        if len(sentences) <= _EXTRACTIVE_MIN:
            cleaned = [_strip_citation_brackets(s) for s in sentences]
            cleaned = [s for s in cleaned if s]
            summary = " ".join(cleaned)
            return summary, len(cleaned)

        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf = vectorizer.fit_transform(sentences)
            scores = tfidf.sum(axis=1).A.flatten()
        except Exception as exc:
            logger.warning("TF-IDF summarization failed: %s", exc)
            picked_raw = sentences[:_EXTRACTIVE_MAX]
            cleaned = [_strip_citation_brackets(s) for s in picked_raw]
            cleaned = [s for s in cleaned if s]
            summary = " ".join(cleaned)
            return summary, len(cleaned)

        n_pick = min(_EXTRACTIVE_MAX, max(_EXTRACTIVE_MIN, len(sentences) // 4))
        n_pick = min(n_pick, len(sentences))
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[
            :n_pick
        ]
        ranked = sorted(ranked)
        picked = [_strip_citation_brackets(sentences[i]) for i in ranked]
        picked = [s for s in picked if s]
        summary = " ".join(picked)
        return summary, len(picked)

    @staticmethod
    def _count_summary_sentences(summary: str) -> int:
        parts = [p for p in re.split(r"(?<=[.!?])\s+", summary.strip()) if p.strip()]
        return len(parts) if parts else (1 if summary.strip() else 0)

    def summarize(self, text: str) -> SummaryResult:
        """
        Return abstractive summary via OpenAI when possible, else extractive TF-IDF.
        """
        stripped = (text or "").strip()
        if not stripped:
            return SummaryResult(summary="", method="extractive", sentence_count=0)

        openai_text = self._summarize_openai(stripped)
        if openai_text:
            return SummaryResult(
                summary=openai_text,
                method="openai",
                sentence_count=self._count_summary_sentences(openai_text),
            )

        summary, n = self._summarize_extractive(stripped)
        return SummaryResult(
            summary=summary,
            method="extractive",
            sentence_count=n,
        )
