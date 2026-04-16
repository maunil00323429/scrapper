"""
Tests for TextSummarizer (OpenAI mock + extractive fallback).
"""

from unittest.mock import MagicMock, patch

from src.analysis.summarizer import SummaryResult, TextSummarizer


LONG_TEXT = (
    "Machine learning is a branch of artificial intelligence. "
    "It uses data to train models that make predictions. "
    "Deep learning uses neural networks with many layers. "
    "Applications include vision, speech, and language. "
    "Researchers continue to improve efficiency and robustness."
)


class TestTextSummarizerExtractive:
    """Extractive path when OpenAI is unavailable."""

    def setup_method(self):
        self.summarizer = TextSummarizer()

    def test_empty_text(self):
        r = self.summarizer.summarize("")
        assert isinstance(r, SummaryResult)
        assert r.summary == ""
        assert r.method == "extractive"
        assert r.sentence_count == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_fallback_extractive_without_key(self):
        r = self.summarizer.summarize(LONG_TEXT)
        assert r.method == "extractive"
        assert len(r.summary) > 0
        assert 3 <= r.sentence_count <= 5

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_extractive_excludes_citation_sentences(self):
        text = (
            "Machine learning is a branch of artificial intelligence work. "
            "It uses data to train models that make predictions today. "
            "Deep learning uses neural networks with many layers here. "
            "This line should be dropped due to cite markers [83] and noise. "
            "Applications include vision speech and language tasks now. "
            "Researchers continue to improve efficiency and robustness always."
        )
        r = self.summarizer.summarize(text)
        assert r.method == "extractive"
        assert "[" not in r.summary and "]" not in r.summary
        assert "cite markers" not in r.summary.lower()

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_extractive_excludes_comma_list_like_sentence(self):
        text = (
            "Machine learning is a branch of artificial intelligence work. "
            "Alice, Bob, Carol, Dave, Erin, Frank, Grace, Henry live here. "
            "Deep learning uses neural networks with many layers here. "
            "Applications include vision speech and language tasks now. "
            "Researchers continue to improve efficiency and robustness always."
        )
        r = self.summarizer.summarize(text)
        assert "Alice" not in r.summary

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_extractive_drops_short_sentences(self):
        text = (
            "This is a sufficiently long first sentence for the test. "
            "Short. "
            "Another sufficiently long second sentence for the test. "
            "Third long sentence about topics and content here now. "
            "Fourth long sentence about more content and ideas here."
        )
        r = self.summarizer.summarize(text)
        assert "Short." not in r.summary

    def test_to_dict(self):
        r = SummaryResult(summary="Hello.", method="extractive", sentence_count=1)
        d = r.to_dict()
        assert d == {"summary": "Hello.", "method": "extractive", "sentence_count": 1}


class TestTextSummarizerOpenAI:
    """OpenAI path with mocked client."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("src.analysis.summarizer.OpenAI")
    def test_openai_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        msg = MagicMock()
        msg.content = (
            "This is one sentence. Here is a second. "
            "A third sentence wraps up."
        )
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )

        s = TextSummarizer()
        r = s.summarize(LONG_TEXT)
        assert r.method == "openai"
        assert "sentence" in r.summary.lower()
        assert r.sentence_count >= 1
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("src.analysis.summarizer.OpenAI")
    def test_openai_failure_falls_back(self, mock_openai_cls):
        mock_openai_cls.side_effect = RuntimeError("API down")

        s = TextSummarizer()
        r = s.summarize(LONG_TEXT)
        assert r.method == "extractive"
        assert len(r.summary) > 0
