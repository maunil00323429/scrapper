# 🔍 Web Content Scraper & Text Analyzer

An NLP-powered system that extracts meaningful text from public web pages and analyzes it for **keywords**, **topics**, **readability**, **sentiment**, and **summaries**, with optional **multi-URL comparison** and **Q&A** over page content.

**GitHub Repository:** [https://github.com/maunil00323429/scrapper](https://github.com/maunil00323429/scrapper)

**Team:** Sudbury AI Solutions (DataScrapers)

---

## Table of Contents

- [Purpose](#purpose)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Project Structure](#project-structure)

---

## Purpose

Many websites contain long articles and documentation that are difficult to analyze manually. Although the content is public, it is often mixed with ads, menus, and irrelevant text. This project extracts meaningful text from public web pages and applies NLP techniques to analyze it, providing structured insights on key topics, keywords, and readability.

## Features

- **Web Content Extraction** — Scrapes and cleans web pages using trafilatura with BeautifulSoup fallback, stripping ads, navigation, and scripts.
- **Keyword Extraction** — Identifies important terms using TF-IDF scoring, Named Entity Recognition, and noun phrase analysis.
- **Topic Detection** — Discovers latent topics using LDA (Latent Dirichlet Allocation).
- **Readability Analysis** — Computes Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau, and other readability metrics.
- **Sentiment (VADER)** — NLTK VADER scores for full text and per-sentence distribution (positive / neutral / negative).
- **Summarization** — OpenAI `gpt-3.5-turbo` abstractive summary when `OPENAI_API_KEY` is set; otherwise TF-IDF extractive top sentences.
- **Comparative Analysis** — Compare 2–3 URLs: readability, sentiment, shared vs unique keywords, topic overlap (`POST /compare` and Streamlit section). Implemented in `src/analysis/comparator.py` (optional module; removing that file disables `/compare` only).
- **Visualizations** — Streamlit: Plotly radar chart for readability, pie chart for sentiment mix, word cloud for TF-IDF keywords, grouped bars for comparisons.
- **Q&A Chatbot** — After analysis, ask questions about the scraped text via OpenAI (`POST /chat` and Streamlit). The UI opens a **dialog** (“Ask” / sidebar) that stays open across follow-up messages; answers use the extracted text and page title as context (long pages are truncated with a clear note). Requires `OPENAI_API_KEY`.
- **Named Entity Recognition** — Identifies people, organizations, locations, dates, and other entities using spaCy.
- **REST API** — FastAPI backend with Swagger UI documentation at `/docs`.
- **Streamlit UI** — Interactive web interface for analysis, comparison, and chat.

## Architecture

```
User → Streamlit UI / API Request
         │
         ▼
   ┌─────────────┐
   │   Scraper    │  ← trafilatura + BeautifulSoup fallback
   │   Module     │
   └──────┬──────┘
          │ clean text
          ▼
   ┌─────────────┐
   │     NLP      │  ← spaCy pipeline (tokenize, NER, POS)
   │  Processor   │
   └──────┬──────┘
          │ processed data
          ▼
   ┌──────────────────────────────────────┐
   │          Analysis Modules            │
   │  ┌──────────┬──────────┬───────────┐ │
   │  │ Keywords │ Topics   │Readability│ │
   │  │ (TF-IDF) │ (LDA)   │(textstat) │ │
   │  └──────────┴──────────┴───────────┘ │
   └──────────────┬───────────────────────┘
                  │
                  ▼
         Structured JSON Response
```

## Tech Stack

| Component           | Technology                        |
|---------------------|-----------------------------------|
| Language            | Python 3.10+                      |
| Web Scraping        | trafilatura, BeautifulSoup, requests |
| NLP Processing      | spaCy (en_core_web_sm)            |
| Keyword Extraction  | scikit-learn (TF-IDF)             |
| Topic Detection     | scikit-learn (LDA)                |
| Readability         | textstat                          |
| Sentiment           | NLTK (VADER)                      |
| Summarization / Q&A| OpenAI (`gpt-3.5-turbo`)          |
| Text Processing     | NLTK                              |
| Visualizations      | Plotly, matplotlib, wordcloud     |
| Configuration       | python-dotenv                     |
| REST API            | FastAPI + Uvicorn                 |
| UI                  | Streamlit (≥ 1.39; dialogs for Q&A) |
| Testing             | pytest                            |
| Containerization    | Docker + Docker Compose           |

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/maunil00323429/scrapper.git
cd scrapper
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment variables

Copy `.env.example` to `.env` in the project root and set your key (`.env` is gitignored and must not be committed):

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell or CMD)
copy .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY` to your key.

Set `OPENAI_API_KEY` to your OpenAI key to enable abstractive summarization and Q&A. If it is left blank, summarization uses the extractive fallback and the UI explains that chat requires a key.

`python-dotenv` loads `.env` when you run the API or Streamlit app.

### Step 5: Download NLP Models & Data

```bash
# spaCy English model
python -m spacy download en_core_web_sm

# NLTK data (includes VADER lexicon used on first run if missing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('vader_lexicon')"
```

## Running the Application

### Option 1: Streamlit UI (Recommended for Demo)

```bash
streamlit run ui/app.py
```

Open your browser at **http://localhost:8501**

### Option 2: FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

- Swagger UI: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

### Option 3: Both (Two Terminals)

**Terminal 1 — API:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — UI:**
```bash
streamlit run ui/app.py
```

## API Documentation

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### `POST /analyze`

Analyze a web page URL.

**Request Body:**
```json
{
  "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
  "num_topics": 3,
  "top_keywords": 15
}
```

**Response:** JSON with `metadata`, `readability`, `keywords`, `topics`, `sentiment`, `summary`, and `entity_summary`.

### `POST /compare`

Analyze **2–3** URLs with the same optional `num_topics` / `top_keywords` as `/analyze`.

**Request body:** `{ "urls": ["https://...", "https://..."], "num_topics": 3, "top_keywords": 15 }`

**Response:** `analyses` (full results per URL) and `comparison` (readability rows, sentiment rows, common/unique keywords, topic overlap). This route is registered only when `src/analysis/comparator.py` is present.

### `POST /chat`

Ask a question about a page’s content. The server fetches the URL, then calls OpenAI with the extracted text and **page title** as structured context (same framing as the Streamlit chatbot).

**Request body:** `{ "url": "https://...", "question": "...", "conversation_history": [{ "role": "user", "content": "..." }] }`

**Response:** `{ "answer": "..." }`. Returns **503** if `OPENAI_API_KEY` is not set or the model call fails in a way surfaced as configuration/service error.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scraper.py -v

# Run with coverage
pytest tests/ -v --cov=src
```

## Docker

### Build and Run with Docker Compose

```bash
# Build and start both services
docker-compose up --build

# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

### Build and Run API Only

```bash
docker build -t web-analyzer .
docker run -p 8000:8000 web-analyzer
```

## Project Structure

```
scrapper/
├── src/
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   └── extractor.py          # Web content extraction (trafilatura + BS4)
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── processor.py          # spaCy NLP pipeline
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── keywords.py           # TF-IDF keyword extraction
│   │   ├── topics.py             # LDA topic detection
│   │   ├── readability.py      # Readability metrics (textstat)
│   │   ├── sentiment.py        # VADER sentiment
│   │   ├── summarizer.py       # OpenAI + extractive summary
│   │   ├── comparator.py     # Multi-URL comparison (optional)
│   │   └── chatbot.py          # OpenAI Q&A over page text
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI application & endpoints
│       └── schemas.py            # Pydantic request/response models
├── ui/
│   └── app.py                    # Streamlit frontend
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py           # Scraper unit tests
│   ├── test_nlp.py               # NLP processor tests
│   ├── test_analysis.py          # Analysis module tests
│   ├── test_summarizer.py
│   ├── test_comparator.py
│   ├── test_chatbot.py
│   └── test_api.py               # API endpoint tests
├── .env.example                  # Template for OPENAI_API_KEY (safe to commit)
├── .env                          # Your secrets — create locally; never commit
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

**Course:** Natural Language Processing — Cambrian College  
**Program:** Graduate Certificate in Artificial Intelligence
