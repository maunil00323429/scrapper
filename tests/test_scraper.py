"""
Tests for the Web Content Extractor module.

Uses mocked HTTP responses to test extraction logic
without making real network requests.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.scraper.extractor import WebContentExtractor, ExtractedContent


# ── Sample HTML Fixtures ────────────────────────────────────────

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Article Title</title></head>
<body>
    <nav><a href="/">Home</a><a href="/about">About</a></nav>
    <script>var tracking = true;</script>
    <article>
        <h1>Test Article Title</h1>
        <p>This is the first paragraph of the test article. It contains
        enough text to pass the minimum content threshold for extraction.
        Natural language processing is a subfield of linguistics and
        computer science concerned with interactions between computers
        and human language.</p>
        <p>The second paragraph discusses machine learning approaches
        to NLP problems including text classification, named entity
        recognition, and sentiment analysis. These techniques have
        revolutionized how we process and understand text data.</p>
        <p>Deep learning models like transformers have achieved
        state-of-the-art results on many NLP benchmarks. BERT, GPT,
        and other large language models continue to push the boundaries
        of what is possible with natural language understanding.</p>
    </article>
    <footer><p>Copyright 2024</p></footer>
</body>
</html>
"""

MINIMAL_HTML = """
<html><head><title>Short</title></head>
<body><p>Too short.</p></body></html>
"""


# ── ExtractedContent Tests ──────────────────────────────────────

class TestExtractedContent:
    """Tests for the ExtractedContent data class."""

    def test_valid_content(self):
        """Content with sufficient text should be valid."""
        content = ExtractedContent(
            url="https://example.com",
            title="Test",
            text="A" * 100,
            author=None,
            date=None,
            word_count=20,
            extraction_method="test",
        )
        assert content.is_valid() is True

    def test_invalid_short_content(self):
        """Content with insufficient text should be invalid."""
        content = ExtractedContent(
            url="https://example.com",
            title="Test",
            text="Short text",
            author=None,
            date=None,
            word_count=2,
            extraction_method="test",
        )
        assert content.is_valid() is False

    def test_empty_content(self):
        """Empty text should be invalid."""
        content = ExtractedContent(
            url="https://example.com",
            title=None,
            text="",
            author=None,
            date=None,
            word_count=0,
            extraction_method="test",
        )
        assert content.is_valid() is False


# ── WebContentExtractor Tests ───────────────────────────────────

class TestWebContentExtractor:
    """Tests for the WebContentExtractor class."""

    def setup_method(self):
        """Initialize extractor for each test."""
        self.extractor = WebContentExtractor(timeout=10)

    @patch("src.scraper.extractor.requests.get")
    def test_fetch_html_success(self, mock_get):
        """Successful HTTP fetch should return HTML string."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        html = self.extractor._fetch_html("https://example.com")
        assert "Test Article Title" in html
        mock_get.assert_called_once()

    @patch("src.scraper.extractor.requests.get")
    def test_fetch_html_failure(self, mock_get):
        """HTTP error should raise RequestException."""
        mock_get.side_effect = Exception("Connection refused")

        with pytest.raises(Exception):
            self.extractor._fetch_html("https://nonexistent.example.com")

    def test_beautifulsoup_extraction(self):
        """BeautifulSoup fallback should extract paragraph content."""
        result = self.extractor._extract_with_beautifulsoup(
            SAMPLE_HTML, "https://example.com"
        )

        assert result is not None
        assert result.title == "Test Article Title"
        assert result.extraction_method == "beautifulsoup"
        assert result.word_count > 10
        assert "natural language processing" in result.text.lower()

    def test_beautifulsoup_removes_nav_and_scripts(self):
        """BeautifulSoup should strip nav, script, and footer elements."""
        result = self.extractor._extract_with_beautifulsoup(
            SAMPLE_HTML, "https://example.com"
        )

        assert result is not None
        assert "Home" not in result.text
        assert "tracking" not in result.text

    def test_beautifulsoup_rejects_minimal_content(self):
        """Pages with too little content should return None."""
        result = self.extractor._extract_with_beautifulsoup(
            MINIMAL_HTML, "https://example.com"
        )
        assert result is None

    @patch("src.scraper.extractor.requests.get")
    def test_extract_raises_on_no_content(self, mock_get):
        """Extract should raise ValueError when no content is found."""
        mock_response = MagicMock()
        mock_response.text = MINIMAL_HTML
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Could not extract"):
            self.extractor.extract("https://example.com/empty")

    def test_custom_timeout_and_user_agent(self):
        """Constructor should accept custom timeout and user agent."""
        extractor = WebContentExtractor(
            timeout=30, user_agent="CustomBot/1.0"
        )
        assert extractor.timeout == 30
        assert extractor.user_agent == "CustomBot/1.0"
