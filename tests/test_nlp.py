"""
Tests for the NLP Processor module.

Tests tokenization, NER, POS tagging, lemmatization,
and entity summary functionality.
"""

import pytest

from src.nlp.processor import NLPProcessor, ProcessedText


# ── Sample Text Fixtures ────────────────────────────────────────

SAMPLE_TEXT = (
    "Natural language processing is a subfield of artificial intelligence. "
    "Google and Microsoft have invested heavily in NLP research. "
    "Transformer models like BERT were introduced in 2018. "
    "These models process text using attention mechanisms. "
    "Applications include machine translation and sentiment analysis."
)

SHORT_TEXT = "Hello world."


# ── NLPProcessor Tests ──────────────────────────────────────────

class TestNLPProcessor:
    """Tests for the NLPProcessor class."""

    @classmethod
    def setup_class(cls):
        """Load spaCy model once for all tests in this class."""
        cls.processor = NLPProcessor(model_name="en_core_web_sm")

    def test_process_returns_processed_text(self):
        """Processing text should return a ProcessedText instance."""
        result = self.processor.process(SAMPLE_TEXT)
        assert isinstance(result, ProcessedText)

    def test_token_count(self):
        """Token count should be positive for non-empty text."""
        result = self.processor.process(SAMPLE_TEXT)
        assert result.token_count > 0

    def test_sentence_segmentation(self):
        """Should correctly segment text into sentences."""
        result = self.processor.process(SAMPLE_TEXT)
        assert result.sentence_count == 5

    def test_unique_tokens_less_than_total(self):
        """Unique token count should not exceed total token count."""
        result = self.processor.process(SAMPLE_TEXT)
        assert result.unique_token_count <= result.token_count

    def test_named_entities_detected(self):
        """Should detect named entities like organizations."""
        result = self.processor.process(SAMPLE_TEXT)
        entity_texts = [e["text"] for e in result.entities]
        # Google or Microsoft should be detected
        assert any(
            name in entity_texts
            for name in ["Google", "Microsoft", "BERT"]
        )

    def test_lemmas_are_lowercase(self):
        """Lemmas should be normalized to lowercase."""
        result = self.processor.process(SAMPLE_TEXT)
        for lemma in result.lemmas:
            assert lemma == lemma.lower()

    def test_noun_chunks_extracted(self):
        """Should extract noun chunks from the text."""
        result = self.processor.process(SAMPLE_TEXT)
        assert len(result.noun_chunks) > 0

    def test_pos_tags_present(self):
        """POS tags should be assigned to tokens."""
        result = self.processor.process(SAMPLE_TEXT)
        assert len(result.pos_tags) > 0
        assert "pos" in result.pos_tags[0]
        assert "token" in result.pos_tags[0]

    def test_max_length_truncation(self):
        """Text should be truncated when max_length is set."""
        long_text = SAMPLE_TEXT * 10
        result_full = self.processor.process(long_text)
        result_truncated = self.processor.process(long_text, max_length=100)
        assert result_truncated.token_count < result_full.token_count

    def test_to_dict(self):
        """to_dict should return a dictionary with expected keys."""
        result = self.processor.process(SAMPLE_TEXT)
        result_dict = result.to_dict()
        assert "sentence_count" in result_dict
        assert "token_count" in result_dict
        assert "entities" in result_dict
        assert "pos_distribution" in result_dict

    def test_short_text_processing(self):
        """Short text should process without errors."""
        result = self.processor.process(SHORT_TEXT)
        assert result.sentence_count >= 1
        assert result.token_count >= 1


class TestEntitySummary:
    """Tests for the entity summary functionality."""

    @classmethod
    def setup_class(cls):
        cls.processor = NLPProcessor(model_name="en_core_web_sm")

    def test_entity_summary_groups_by_label(self):
        """Entity summary should group entities by their NER label."""
        result = self.processor.process(SAMPLE_TEXT)
        summary = self.processor.get_entity_summary(result)
        assert isinstance(summary, dict)
        # All values should be lists
        for label, entities in summary.items():
            assert isinstance(entities, list)

    def test_entity_summary_deduplicates(self):
        """Entity summary should not contain duplicate entities."""
        repeated_text = (
            "Google released a new product. Google announced earnings. "
            "Google is based in Mountain View."
        )
        result = self.processor.process(repeated_text)
        summary = self.processor.get_entity_summary(result)
        for label, entities in summary.items():
            assert len(entities) == len(set(entities))
