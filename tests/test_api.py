"""
Tests for the FastAPI API endpoints.

Uses httpx AsyncClient to test API routes
with mocked processing components.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self):
        """Health endpoint should return 200 with status healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_returns_version(self):
        """Health endpoint should include the API version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    def test_analyze_requires_url(self):
        """Analyze should reject requests without a URL."""
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_analyze_rejects_invalid_url(self):
        """Analyze should reject malformed URLs."""
        response = client.post("/analyze", json={"url": "not-a-url"})
        assert response.status_code == 422

    def test_analyze_rejects_unreachable_url(self):
        """Analyze should return 400 for unreachable URLs."""
        response = client.post(
            "/analyze",
            json={"url": "https://thisdomaindoesnotexist12345.com/page"},
        )
        assert response.status_code == 400


class TestCompareEndpoint:
    """Tests for POST /compare."""

    def test_compare_requires_two_urls(self):
        response = client.post(
            "/compare",
            json={"urls": ["https://example.com"]},
        )
        assert response.status_code == 422


class TestChatEndpoint:
    """Tests for POST /chat."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    @patch("src.api.main.scraper.extract")
    def test_chat_503_without_api_key(self, mock_extract):
        mock_extract.return_value = MagicMock(
            text="Hello world content about NLP.",
            url="https://example.com",
            title="Hi",
            author=None,
            date=None,
            word_count=5,
            extraction_method="test",
        )
        response = client.post(
            "/chat",
            json={"url": "https://example.com", "question": "What is this about?"},
        )
        assert response.status_code == 503


class TestSwaggerDocs:
    """Tests for API documentation endpoints."""

    def test_swagger_docs_available(self):
        """Swagger UI should be accessible at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """ReDoc should be accessible at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """OpenAPI JSON schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Web Content Scraper & Text Analyzer"
        paths = data.get("paths", {})
        assert "/compare" in paths
        assert "/chat" in paths
