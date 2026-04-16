"""Tests for ContentComparator."""

import pytest

from src.analysis.comparator import ComparisonResult, ContentComparator


def _fake_analysis(url: str, kw_terms: list[str], topic_words: list[str]) -> dict:
    return {
        "metadata": {
            "url": url,
            "title": "T",
            "author": None,
            "date": None,
            "word_count": 100,
            "extraction_method": "test",
        },
        "readability": {
            "scores": {
                "flesch_reading_ease": 60.0,
                "flesch_kincaid_grade": 8.0,
                "gunning_fog": 10.0,
                "smog_index": 9.0,
                "coleman_liau_index": 10.0,
                "automated_readability_index": 9.0,
                "dale_chall_score": 8.0,
            },
            "reading_level": "Standard",
            "estimated_reading_time_minutes": 1.0,
            "text_statistics": {
                "character_count": 500,
                "word_count": 100,
                "sentence_count": 5,
                "syllable_count": 150,
                "avg_sentence_length": 20.0,
                "avg_word_length": 5.0,
                "avg_syllables_per_word": 1.5,
            },
        },
        "keywords": {
            "tfidf_keywords": [{"keyword": k, "score": 1.0} for k in kw_terms],
            "entity_keywords": [],
            "noun_phrase_keywords": [],
        },
        "topics": {
            "lda_topics": [
                {
                    "topic_id": 1,
                    "words": topic_words,
                    "weights": [0.1] * len(topic_words),
                    "label": "Topic 1",
                }
            ]
        },
        "sentiment": {
            "overall": {
                "compound": 0.1,
                "positive": 0.2,
                "negative": 0.0,
                "neutral": 0.8,
                "label": "neutral",
            },
            "sentence_sentiments": [],
            "distribution": {"positive": 1, "negative": 0, "neutral": 2},
        },
        "summary": {"summary": "x", "method": "extractive", "sentence_count": 1},
        "entity_summary": {},
    }


def test_compare_finds_common_keyword():
    c = ContentComparator()
    a1 = _fake_analysis("https://a.example", ["apple", "banana"], ["fruit", "food"])
    a2 = _fake_analysis("https://b.example", ["banana", "cherry"], ["fruit", "taste"])
    r = c.compare([a1, a2])
    assert isinstance(r, ComparisonResult)
    assert "banana" in r.common_keywords
    d = r.to_dict()
    assert len(d["urls"]) == 2
    assert len(d["readability"]) == 2
    assert "topic_overlap" in d


def test_compare_requires_two():
    c = ContentComparator()
    with pytest.raises(ValueError):
        c.compare([_fake_analysis("https://a.example", ["x"], ["y"])])
