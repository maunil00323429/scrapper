"""
API Schema Definitions

Pydantic models for request validation and response serialization
used by the FastAPI endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    url: HttpUrl = Field(
        ...,
        description="Public web page URL to scrape and analyze",
        examples=["https://en.wikipedia.org/wiki/Natural_language_processing"],
    )
    num_topics: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of topics to detect",
    )
    top_keywords: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of top keywords to extract",
    )


class TextStatistics(BaseModel):
    """Basic text statistics."""

    character_count: int
    word_count: int
    sentence_count: int
    syllable_count: int
    avg_sentence_length: float
    avg_word_length: float
    avg_syllables_per_word: float


class ReadabilityScores(BaseModel):
    """Readability metric scores."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    dale_chall_score: float


class ReadabilityResponse(BaseModel):
    """Readability analysis results."""

    scores: ReadabilityScores
    reading_level: str
    estimated_reading_time_minutes: float
    text_statistics: TextStatistics


class KeywordItem(BaseModel):
    """Single TF-IDF keyword with score."""

    keyword: str
    score: float


class EntityItem(BaseModel):
    """Single named entity."""

    text: str
    label: str


class KeywordsResponse(BaseModel):
    """Keyword extraction results."""

    tfidf_keywords: list[KeywordItem]
    entity_keywords: list[EntityItem]
    noun_phrase_keywords: list[str]


class TopicItem(BaseModel):
    """Single detected topic."""

    topic_id: int
    words: list[str]
    weights: list[float]
    label: str


class TopicsResponse(BaseModel):
    """Topic detection results."""

    lda_topics: list[TopicItem]


class SentimentOverall(BaseModel):
    """Aggregate VADER scores for the full text."""

    compound: float
    positive: float
    negative: float
    neutral: float
    label: str


class SentenceSentimentItem(BaseModel):
    """VADER scores for a single sentence."""

    sentence: str
    compound: float
    positive: float
    negative: float
    neutral: float
    label: str


class SentimentDistribution(BaseModel):
    """Counts of sentences by sentiment label."""

    positive: int
    negative: int
    neutral: int


class SentimentResponse(BaseModel):
    """Sentiment analysis results."""

    overall: SentimentOverall
    sentence_sentiments: list[SentenceSentimentItem]
    distribution: SentimentDistribution


class SummaryResponse(BaseModel):
    """Text summary from OpenAI or extractive fallback."""

    summary: str
    method: str
    sentence_count: int


class ContentMetadata(BaseModel):
    """Extracted web page metadata."""

    url: str
    title: Optional[str]
    author: Optional[str]
    date: Optional[str]
    word_count: int
    extraction_method: str


class AnalyzeResponse(BaseModel):
    """Complete analysis response combining all modules."""

    metadata: ContentMetadata
    readability: ReadabilityResponse
    keywords: KeywordsResponse
    topics: TopicsResponse
    sentiment: SentimentResponse
    summary: SummaryResponse
    entity_summary: dict[str, list[str]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class CompareRequest(BaseModel):
    """Request body for POST /compare."""

    urls: list[HttpUrl] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Two or three URLs to analyze and compare",
    )
    num_topics: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of topics to detect per URL",
    )
    top_keywords: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Top keywords per URL",
    )


class CompareResponse(BaseModel):
    """Full analyses plus structured comparison."""

    analyses: list[AnalyzeResponse]
    comparison: dict


class ChatMessageItem(BaseModel):
    """Single turn in chat history."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    url: HttpUrl
    question: str
    conversation_history: list[ChatMessageItem] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Assistant reply for Q&A."""

    answer: str
