"""
NLP Processor Module

Provides core NLP processing capabilities using spaCy, including
tokenization, named entity recognition, POS tagging, and sentence
segmentation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

# Default spaCy model
DEFAULT_MODEL: str = "en_core_web_sm"


@dataclass
class ProcessedText:
    """Data class holding NLP-processed text results."""

    tokens: list[str]
    sentences: list[str]
    entities: list[dict[str, str]]
    pos_tags: list[dict[str, str]]
    lemmas: list[str]
    noun_chunks: list[str]
    sentence_count: int
    token_count: int
    unique_token_count: int

    def to_dict(self) -> dict:
        """Convert processed text results to a dictionary."""
        return {
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "unique_token_count": self.unique_token_count,
            "entities": self.entities,
            "noun_chunks": self.noun_chunks[:20],
            "pos_distribution": self._get_pos_distribution(),
        }

    def _get_pos_distribution(self) -> dict[str, int]:
        """Calculate the distribution of POS tags."""
        distribution: dict[str, int] = {}
        for item in self.pos_tags:
            tag = item["pos"]
            distribution[tag] = distribution.get(tag, 0) + 1
        return dict(
            sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        )


class NLPProcessor:
    """
    Core NLP processing engine powered by spaCy.

    Handles tokenization, named entity recognition, POS tagging,
    lemmatization, sentence segmentation, and noun chunk extraction.

    Attributes:
        model_name: Name of the spaCy language model to use.
        nlp: Loaded spaCy language model pipeline.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize the NLP processor with a spaCy model.

        Args:
            model_name: spaCy model identifier (e.g., 'en_core_web_sm').

        Raises:
            OSError: If the specified spaCy model is not installed.
        """
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. "
                "Install it with: python -m spacy download %s",
                model_name,
                model_name,
            )
            raise

    def process(self, text: str, max_length: Optional[int] = None) -> ProcessedText:
        """
        Run the full NLP pipeline on the input text.

        Args:
            text: Raw text string to process.
            max_length: Optional character limit for processing.
                        Truncates text if it exceeds this length.

        Returns:
            ProcessedText containing all NLP analysis results.
        """
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.info("Text truncated to %d characters", max_length)

        doc: Doc = self.nlp(text)

        # Extract tokens (excluding punctuation and whitespace)
        tokens = [
            token.text for token in doc
            if not token.is_punct and not token.is_space
        ]

        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]

        # Extract named entities
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

        # Extract POS tags
        pos_tags = [
            {"token": token.text, "pos": token.pos_, "tag": token.tag_}
            for token in doc
            if not token.is_punct and not token.is_space
        ]

        # Extract lemmas (excluding stopwords, punct, whitespace)
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text) > 1
        ]

        # Extract noun chunks
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        return ProcessedText(
            tokens=tokens,
            sentences=sentences,
            entities=entities,
            pos_tags=pos_tags,
            lemmas=lemmas,
            noun_chunks=noun_chunks,
            sentence_count=len(sentences),
            token_count=len(tokens),
            unique_token_count=len(set(tokens)),
        )

    def get_entity_summary(self, processed: ProcessedText) -> dict[str, list[str]]:
        """
        Group named entities by their category labels.

        Args:
            processed: Previously processed NLP results.

        Returns:
            Dictionary mapping entity labels to lists of unique entity texts.
        """
        summary: dict[str, list[str]] = {}
        for entity in processed.entities:
            label = entity["label"]
            text = entity["text"]
            if label not in summary:
                summary[label] = []
            if text not in summary[label]:
                summary[label].append(text)
        return summary
