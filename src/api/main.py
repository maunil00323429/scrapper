"""
FastAPI Application

REST API for the Web Content Scraper & Text Analyzer.
Provides endpoints for analyzing web page content with NLP techniques
including keyword extraction, topic detection, and readability analysis.

Run with: uvicorn src.api.main:app --reload
Swagger docs available at: http://localhost:8000/docs
"""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
)
from src.scraper.extractor import WebContentExtractor
from src.nlp.processor import NLPProcessor
from src.analysis.keywords import KeywordExtractor
from src.analysis.topics import TopicDetector
from src.analysis.readability import ReadabilityAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.summarizer import TextSummarizer
from src.analysis.chatbot import ContentChatbot

load_dotenv()

# Optional comparator: deleting src/analysis/comparator.py removes /compare only.
try:
    from src.analysis.comparator import ContentComparator
except ImportError:  # pragma: no cover
    ContentComparator = None  # type: ignore[misc, assignment]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "Web Content Scraper & Text Analyzer"
APP_DESCRIPTION = (
    "An NLP-powered API that extracts meaningful text from public web pages "
    "and analyzes it for keywords, topics, and readability. "
    "Submit a URL to receive structured analysis results."
)
APP_VERSION = "1.0.0"

# Initialize FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing components (loaded once at startup)
scraper = WebContentExtractor()
nlp_processor = NLPProcessor()
readability_analyzer = ReadabilityAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
text_summarizer = TextSummarizer()

if ContentComparator is not None:
    comparator = ContentComparator()
else:  # pragma: no cover
    comparator = None


def _perform_analysis_sync(
    url: str, num_topics: int, top_keywords: int
) -> AnalyzeResponse:
    """Run the full analysis pipeline; raises on failure (no HTTP)."""
    logger.info("Analyzing URL: %s", url)

    content = scraper.extract(url)
    processed = nlp_processor.process(content.text)

    kw_extractor = KeywordExtractor(top_n=top_keywords)
    keywords = kw_extractor.extract(processed)

    t_detector = TopicDetector(num_topics=num_topics)
    topics = t_detector.detect(processed.sentences)

    readability = readability_analyzer.analyze(content.text)
    sentiment = sentiment_analyzer.analyze(content.text)
    summary = text_summarizer.summarize(content.text)
    entity_summary = nlp_processor.get_entity_summary(processed)

    return AnalyzeResponse(
        metadata={
            "url": content.url,
            "title": content.title,
            "author": content.author,
            "date": content.date,
            "word_count": content.word_count,
            "extraction_method": content.extraction_method,
        },
        readability=readability.to_dict(),
        keywords=keywords.to_dict(),
        topics=topics.to_dict(),
        sentiment=sentiment.to_dict(),
        summary=summary.to_dict(),
        entity_summary=entity_summary,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the current API status and version number.
    """
    return HealthResponse(status="healthy", version=APP_VERSION)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_url(request: AnalyzeRequest):
    """
    Analyze a web page URL.

    Scrapes the given URL, extracts clean text content, and performs
    NLP analysis including keyword extraction, topic detection,
    readability scoring, and named entity recognition.

    Args:
        request: AnalyzeRequest containing the target URL and options.

    Returns:
        AnalyzeResponse with complete analysis results.

    Raises:
        HTTPException: 400 if content extraction fails.
        HTTPException: 500 if an internal processing error occurs.
    """
    url = str(request.url)
    try:
        return _perform_analysis_sync(
            url, request.num_topics, request.top_keywords
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", url, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch or extract content from URL: {exc}",
        )


if ContentComparator is not None:

    @app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
    async def compare_urls(request: CompareRequest):
        """
        Analyze 2–3 URLs and return side-by-side comparison metrics.
        """
        analyses: list[AnalyzeResponse] = []
        for u in request.urls:
            ur = str(u)
            try:
                analyses.append(
                    _perform_analysis_sync(
                        ur, request.num_topics, request.top_keywords
                    )
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                logger.error("Compare failed for %s: %s", ur, exc)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to analyze {ur}: {exc}",
                )
        assert comparator is not None
        comparison = comparator.compare([a.model_dump() for a in analyses])
        return CompareResponse(
            analyses=analyses,
            comparison=comparison.to_dict(),
        )


@app.post("/chat", response_model=ChatResponse, tags=["Analysis"])
async def chat_about_url(request: ChatRequest):
    """
    Ask a question about the content at a URL (OpenAI required).
    """
    url = str(request.url)
    try:
        content = scraper.extract(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Scraping failed for chat %s: %s", url, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch or extract content from URL: {exc}",
        )

    bot = ContentChatbot(content.text, page_title=content.title)
    history = [
        {"role": m.role, "content": m.content}
        for m in request.conversation_history
    ]
    try:
        answer = bot.answer(request.question, history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return ChatResponse(answer=answer)
