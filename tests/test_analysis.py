"""
Tests for the Analysis modules.

Covers keyword extraction, topic detection, and readability analysis.
"""

import pytest

from src.nlp.processor import NLPProcessor
from src.analysis.keywords import KeywordExtractor, KeywordResult
from src.analysis.topics import TopicDetector, TopicResult
from src.analysis.readability import ReadabilityAnalyzer, ReadabilityResult
from src.analysis.sentiment import SentimentAnalyzer, SentimentResult


# ── Sample Text ─────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Natural language processing is a subfield of artificial intelligence "
    "that focuses on the interaction between computers and humans using "
    "natural language. The ultimate objective of NLP is to read, decipher, "
    "understand, and make sense of human languages in a manner that is valuable. "
    "Most NLP techniques rely on machine learning to derive meaning from "
    "human languages. NLP combines computational linguistics with statistical, "
    "machine learning, and deep learning models. These technologies enable "
    "computers to process human language in the form of text or voice data. "
    "Applications of NLP include speech recognition, sentiment analysis, "
    "machine translation, text summarization, and chatbot development. "
    "Modern NLP approaches use transformer architectures like BERT and GPT. "
    "These models have significantly improved the state of the art in many "
    "NLP tasks and continue to evolve rapidly."
)


@pytest.fixture(scope="module")
def processed_text():
    """Fixture: NLP-processed version of the sample text."""
    processor = NLPProcessor(model_name="en_core_web_sm")
    return processor.process(SAMPLE_TEXT)


# ── Keyword Extraction Tests ───────────────────────────────────

class TestKeywordExtractor:
    """Tests for the KeywordExtractor class."""

    def test_extract_returns_keyword_result(self, processed_text):
        """Should return a KeywordResult instance."""
        extractor = KeywordExtractor(top_n=10)
        result = extractor.extract(processed_text)
        assert isinstance(result, KeywordResult)

    def test_tfidf_keywords_not_empty(self, processed_text):
        """TF-IDF should produce at least one keyword."""
        extractor = KeywordExtractor(top_n=10)
        result = extractor.extract(processed_text)
        assert len(result.tfidf_keywords) > 0

    def test_tfidf_keywords_have_scores(self, processed_text):
        """Each TF-IDF keyword should have a score."""
        extractor = KeywordExtractor(top_n=5)
        result = extractor.extract(processed_text)
        for kw in result.tfidf_keywords:
            assert "keyword" in kw
            assert "score" in kw
            assert kw["score"] > 0

    def test_top_n_limits_results(self, processed_text):
        """Number of keywords should not exceed top_n."""
        extractor = KeywordExtractor(top_n=3)
        result = extractor.extract(processed_text)
        assert len(result.tfidf_keywords) <= 3

    def test_noun_phrases_extracted(self, processed_text):
        """Should extract noun phrases from the text."""
        extractor = KeywordExtractor(top_n=10)
        result = extractor.extract(processed_text)
        assert len(result.noun_phrase_keywords) > 0

    def test_to_dict(self, processed_text):
        """to_dict should return a dictionary with expected keys."""
        extractor = KeywordExtractor()
        result = extractor.extract(processed_text)
        result_dict = result.to_dict()
        assert "tfidf_keywords" in result_dict
        assert "entity_keywords" in result_dict
        assert "noun_phrase_keywords" in result_dict

    def test_metadata_noise_words_not_in_tfidf(self):
        """Wikipedia-style metadata tokens should not appear in keyword results."""
        wiki_tail = (
            " Retrieved March 2020. External links and references section. "
            "The paper has a doi identifier for citation purposes."
        )
        processor = NLPProcessor(model_name="en_core_web_sm")
        processed = processor.process(SAMPLE_TEXT + wiki_tail)
        extractor = KeywordExtractor(top_n=30)
        result = extractor.extract(processed)
        lowered = [k["keyword"].lower() for k in result.tfidf_keywords]
        for noise in ("retrieved", "references", "doi", "external"):
            assert noise not in lowered
        np_lower = [p.lower() for p in result.noun_phrase_keywords]
        assert "doi" not in np_lower


# ── Topic Detection Tests ──────────────────────────────────────

class TestTopicDetector:
    """Tests for the TopicDetector class."""

    def test_detect_returns_topic_result(self, processed_text):
        """Should return a TopicResult instance."""
        detector = TopicDetector(num_topics=2)
        result = detector.detect(processed_text.sentences)
        assert isinstance(result, TopicResult)

    def test_lda_topics_generated(self, processed_text):
        """LDA should generate topics when given enough sentences."""
        detector = TopicDetector(num_topics=2)
        result = detector.detect(processed_text.sentences)
        assert len(result.lda_topics) > 0

    def test_topic_has_words(self, processed_text):
        """Each topic should contain representative words."""
        detector = TopicDetector(num_topics=2, words_per_topic=5)
        result = detector.detect(processed_text.sentences)
        for topic in result.lda_topics:
            assert len(topic["words"]) > 0
            assert "topic_id" in topic
            assert "label" in topic

    def test_to_dict(self, processed_text):
        """to_dict should return expected structure."""
        detector = TopicDetector(num_topics=2)
        result = detector.detect(processed_text.sentences)
        result_dict = result.to_dict()
        assert "lda_topics" in result_dict
        assert list(result_dict.keys()) == ["lda_topics"]

    def test_single_sentence_returns_empty(self):
        """Single sentence should not produce topics."""
        detector = TopicDetector(num_topics=2)
        result = detector.detect(["Only one sentence here."])
        assert len(result.lda_topics) == 0


# ── Sentiment Analysis Tests ───────────────────────────────────

class TestSentimentAnalyzer:
    """Tests for VADER-based SentimentAnalyzer."""

    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_analyze_returns_result(self):
        """Should return SentimentResult."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, SentimentResult)

    def test_overall_label_in_set(self):
        """Overall label should be positive, negative, or neutral."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert result.label in ("positive", "negative", "neutral")

    def test_compound_range(self):
        """Compound score should be within [-1, 1]."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert -1.0 <= result.compound <= 1.0

    def test_sentence_distribution_sums(self):
        """Distribution counts should match number of scored sentences."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        total = sum(result.distribution.values())
        assert total == len(result.sentence_sentiments)
        assert total > 0

    def test_positive_text_skews_positive(self):
        """Clearly positive text should not be labeled negative overall."""
        text = "I love this! It is amazing, wonderful, and fantastic. Great work!"
        result = self.analyzer.analyze(text)
        assert result.label != "negative"

    def test_empty_text_neutral(self):
        """Empty input yields neutral overall and empty sentences."""
        result = self.analyzer.analyze("")
        assert result.label == "neutral"
        assert result.sentence_sentiments == []
        assert sum(result.distribution.values()) == 0

    def test_to_dict_structure(self):
        """to_dict matches API nesting."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        d = result.to_dict()
        assert "overall" in d
        assert set(d["overall"]) >= {"compound", "positive", "negative", "neutral", "label"}
        assert "sentence_sentiments" in d
        assert "distribution" in d


# ── Readability Analysis Tests ─────────────────────────────────

class TestReadabilityAnalyzer:
    """Tests for the ReadabilityAnalyzer class."""

    def setup_method(self):
        self.analyzer = ReadabilityAnalyzer()

    def test_analyze_returns_result(self):
        """Should return a ReadabilityResult instance."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, ReadabilityResult)

    def test_flesch_reading_ease_range(self):
        """Flesch Reading Ease should be a reasonable number."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert -50 <= result.flesch_reading_ease <= 120

    def test_grade_level_positive(self):
        """Flesch-Kincaid Grade should be a positive number."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert result.flesch_kincaid_grade > 0

    def test_reading_level_assigned(self):
        """A human-readable reading level should be assigned."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert len(result.reading_level) > 0

    def test_reading_time_positive(self):
        """Estimated reading time should be positive."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert result.estimated_reading_time_minutes > 0

    def test_text_statistics(self):
        """Text statistics should contain expected metrics."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        stats = result.text_statistics
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0
        assert stats["character_count"] > 0
        assert stats["avg_sentence_length"] > 0

    def test_to_dict(self):
        """to_dict should return expected structure."""
        result = self.analyzer.analyze(SAMPLE_TEXT)
        result_dict = result.to_dict()
        assert "scores" in result_dict
        assert "reading_level" in result_dict
        assert "text_statistics" in result_dict

    def test_easy_text_reading_level(self):
        """Simple text should get an easy reading level."""
        easy_text = (
            "The cat sat on the mat. The dog ran in the park. "
            "I like to eat food. The sun is hot today. "
            "We play games at home. She reads a big book."
        )
        result = self.analyzer.analyze(easy_text)
        assert "Easy" in result.reading_level or "Very Easy" in result.reading_level
