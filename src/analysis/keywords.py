"""
Keyword Extraction Module

Extracts important keywords and key phrases from text using
TF-IDF scoring combined with spaCy-based NER and noun chunk analysis.
"""

import logging
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp.processor import ProcessedText

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_TOP_N: int = 15
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 3)

# Filtered from final keyword outputs only (Wikipedia / reference noise)
METADATA_NOISE_WORDS: frozenset[str] = frozenset({
    "retrieved",
    "archived",
    "original",
    "accessed",
    "references",
    "external",
    "links",
    "bibliography",
    "citation",
    "cited",
    "isbn",
    "doi",
    "pdf",
    "http",
    "https",
    "www",
    "com",
    "org",
})


def _term_contains_noise_token(term: str) -> bool:
    """True if any whitespace-/hyphen-separated token is a metadata noise word."""
    for raw in re.split(r"[\s\-/]+", term.lower()):
        tok = raw.strip(".,;:!?'\"()[]")
        if tok in METADATA_NOISE_WORDS:
            return True
    return False


NOUN_PHRASE_PRONOUNS: frozenset[str] = frozenset({
    "him",
    "who",
    "he",
    "she",
    "it",
    "they",
    "them",
    "his",
    "her",
    "its",
    "we",
    "us",
})


def _is_valid_noun_phrase(phrase: str) -> bool:
    """Drop symbols-only, too-short, non-alphabetic, and pronoun-only chunks."""
    normalized = phrase.lower().strip()
    if len(normalized) < 2:
        return False
    if not any(c.isalpha() for c in normalized):
        return False
    tokens = re.findall(r"[a-zA-Z']+", normalized)
    if not tokens:
        return False
    if all(t.lower() in NOUN_PHRASE_PRONOUNS for t in tokens):
        return False
    return True


@dataclass
class KeywordResult:
    """Data class holding keyword extraction results."""

    tfidf_keywords: list[dict[str, float]]
    entity_keywords: list[dict[str, str]]
    noun_phrase_keywords: list[str]

    def to_dict(self) -> dict:
        """Convert keyword results to a dictionary."""
        return {
            "tfidf_keywords": self.tfidf_keywords,
            "entity_keywords": self.entity_keywords,
            "noun_phrase_keywords": self.noun_phrase_keywords[:15],
        }


class KeywordExtractor:
    """
    Extracts keywords from text using multiple strategies.

    Combines TF-IDF vectorization for statistical keyword scoring
    with spaCy-derived named entities and noun chunks for
    linguistically-informed keyword extraction.

    Attributes:
        top_n: Number of top keywords to return.
        ngram_range: Range of n-gram sizes for TF-IDF.
    """

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    ) -> None:
        """
        Initialize the keyword extractor.

        Args:
            top_n: Number of top keywords to extract.
            ngram_range: Min and max n-gram sizes for TF-IDF analysis.
        """
        self.top_n = top_n
        self.ngram_range = ngram_range

    def _extract_tfidf_keywords(
        self, sentences: list[str]
    ) -> list[dict[str, float]]:
        """
        Extract keywords using TF-IDF scoring across sentences.

        Each sentence is treated as a separate document to compute
        term frequency-inverse document frequency scores.

        Args:
            sentences: List of sentence strings from the text.

        Returns:
            List of dicts with 'keyword' and 'score' keys,
            sorted by score descending.
        """
        if len(sentences) < 2:
            return []

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=500,
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            logger.warning("TF-IDF vectorization failed (empty vocabulary)")
            return []

        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF scores across all sentences
        avg_scores = tfidf_matrix.mean(axis=0).A1

        # Pair features with scores; take extra candidates then filter noise
        order = avg_scores.argsort()[::-1]
        scored_keywords = [
            {"keyword": feature_names[i], "score": round(float(avg_scores[i]), 4)}
            for i in order[: self.top_n * 6]
            if avg_scores[i] > 0
        ]
        filtered = [
            item
            for item in scored_keywords
            if not _term_contains_noise_token(item["keyword"])
        ]
        return filtered[: self.top_n]

    def _extract_entity_keywords(
        self, processed: ProcessedText
    ) -> list[dict[str, str]]:
        """
        Extract keyword entities from NER results.

        Deduplicates entities and returns unique named entities
        found in the text.

        Args:
            processed: NLP-processed text containing entity data.

        Returns:
            List of unique entity dicts with 'text' and 'label' keys.
        """
        seen = set()
        unique_entities = []
        for entity in processed.entities:
            if _term_contains_noise_token(entity["text"]):
                continue
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        return unique_entities

    def _extract_noun_phrases(
        self, processed: ProcessedText
    ) -> list[str]:
        """
        Extract and rank noun phrases by frequency.

        Args:
            processed: NLP-processed text containing noun chunks.

        Returns:
            List of noun phrases sorted by frequency (most common first).
        """
        phrase_counts: dict[str, int] = {}
        for chunk in processed.noun_chunks:
            normalized = chunk.lower().strip()
            if not _is_valid_noun_phrase(normalized):
                continue
            phrase_counts[normalized] = phrase_counts.get(normalized, 0) + 1

        sorted_phrases = sorted(
            phrase_counts.items(), key=lambda x: x[1], reverse=True
        )
        out: list[str] = []
        for phrase, _ in sorted_phrases:
            if _term_contains_noise_token(phrase):
                continue
            out.append(phrase)
            if len(out) >= self.top_n:
                break
        return out

    def extract(self, processed: ProcessedText) -> KeywordResult:
        """
        Run all keyword extraction strategies on processed text.

        Args:
            processed: NLP-processed text from the NLPProcessor.

        Returns:
            KeywordResult combining TF-IDF, entity, and noun phrase keywords.
        """
        logger.info("Extracting keywords from %d sentences", processed.sentence_count)

        tfidf_keywords = self._extract_tfidf_keywords(processed.sentences)
        entity_keywords = self._extract_entity_keywords(processed)
        noun_phrases = self._extract_noun_phrases(processed)

        return KeywordResult(
            tfidf_keywords=tfidf_keywords,
            entity_keywords=entity_keywords,
            noun_phrase_keywords=noun_phrases,
        )
