"""
Web Content Extractor Module

Extracts clean, meaningful text from web pages by stripping ads,
navigation menus, scripts, and other non-content elements.
Uses trafilatura as the primary extraction engine with BeautifulSoup as fallback.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Request configuration
DEFAULT_TIMEOUT: int = 15
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class ExtractedContent:
    """Data class holding the results of web content extraction."""

    url: str
    title: Optional[str]
    text: str
    author: Optional[str]
    date: Optional[str]
    word_count: int
    extraction_method: str

    def is_valid(self) -> bool:
        """Check if extracted content meets minimum quality threshold."""
        return len(self.text.strip()) > 50


class WebContentExtractor:
    """
    Extracts clean text content from web pages.

    Uses trafilatura as the primary extraction engine for high-quality
    article text extraction. Falls back to BeautifulSoup-based extraction
    when trafilatura cannot process the page.

    Attributes:
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header string for HTTP requests.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        """
        Initialize the extractor with HTTP configuration.

        Args:
            timeout: Request timeout in seconds.
            user_agent: Browser user-agent string for requests.
        """
        self.timeout = timeout
        self.user_agent = user_agent

    def _fetch_html(self, url: str) -> str:
        """
        Fetch raw HTML content from a URL.

        Args:
            url: The target web page URL.

        Returns:
            Raw HTML string.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _extract_with_trafilatura(
        self, html: str, url: str
    ) -> Optional[ExtractedContent]:
        """
        Extract content using trafilatura (primary method).

        Args:
            html: Raw HTML string.
            url: Original URL for metadata.

        Returns:
            ExtractedContent if successful, None otherwise.
        """
        try:
            metadata = trafilatura.extract(
                html,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            if not metadata or len(metadata.strip()) < 50:
                return None

            # Extract metadata separately
            meta = trafilatura.extract(
                html,
                output_format="xml",
                include_comments=False,
            )

            title = None
            author = None
            date = None

            if meta:
                from xml.etree import ElementTree

                try:
                    root = ElementTree.fromstring(meta)
                    title = root.get("title")
                    author = root.get("author")
                    date = root.get("date")
                except ElementTree.ParseError:
                    pass

            # Fallback title from HTML
            if not title:
                soup = BeautifulSoup(html, "lxml")
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else None

            text = metadata.strip()

            return ExtractedContent(
                url=url,
                title=title,
                text=text,
                author=author,
                date=date,
                word_count=len(text.split()),
                extraction_method="trafilatura",
            )

        except Exception as exc:
            logger.warning("Trafilatura extraction failed: %s", exc)
            return None

    def _extract_with_beautifulsoup(
        self, html: str, url: str
    ) -> Optional[ExtractedContent]:
        """
        Extract content using BeautifulSoup (fallback method).

        Removes script, style, nav, header, footer, and aside elements
        before extracting paragraph text.

        Args:
            html: Raw HTML string.
            url: Original URL for metadata.

        Returns:
            ExtractedContent if successful, None otherwise.
        """
        try:
            soup = BeautifulSoup(html, "lxml")

            # Remove non-content elements
            tags_to_remove = [
                "script", "style", "nav", "header", "footer",
                "aside", "form", "iframe", "noscript",
            ]
            for tag_name in tags_to_remove:
                for element in soup.find_all(tag_name):
                    element.decompose()

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None

            # Extract main content from paragraphs
            paragraphs = soup.find_all("p")
            text_parts = []
            for p in paragraphs:
                content = p.get_text(strip=True)
                if len(content) > 30:  # Filter short/noisy paragraphs
                    text_parts.append(content)

            text = "\n\n".join(text_parts)

            if len(text.strip()) < 50:
                return None

            return ExtractedContent(
                url=url,
                title=title,
                text=text,
                author=None,
                date=None,
                word_count=len(text.split()),
                extraction_method="beautifulsoup",
            )

        except Exception as exc:
            logger.warning("BeautifulSoup extraction failed: %s", exc)
            return None

    def extract(self, url: str) -> ExtractedContent:
        """
        Extract clean text content from a web page URL.

        Attempts trafilatura extraction first, falls back to BeautifulSoup
        if the primary method fails or returns insufficient content.

        Args:
            url: The web page URL to extract content from.

        Returns:
            ExtractedContent with the extracted text and metadata.

        Raises:
            ValueError: If no content could be extracted from the URL.
            requests.RequestException: If the page cannot be fetched.
        """
        logger.info("Extracting content from: %s", url)

        html = self._fetch_html(url)

        # Try trafilatura first (higher quality)
        result = self._extract_with_trafilatura(html, url)
        if result and result.is_valid():
            logger.info(
                "Extracted %d words via trafilatura", result.word_count
            )
            return result

        # Fallback to BeautifulSoup
        logger.info("Falling back to BeautifulSoup extraction")
        result = self._extract_with_beautifulsoup(html, url)
        if result and result.is_valid():
            logger.info(
                "Extracted %d words via BeautifulSoup", result.word_count
            )
            return result

        raise ValueError(
            f"Could not extract meaningful content from: {url}"
        )
