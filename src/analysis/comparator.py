"""
Compare analysis results from multiple URLs (standalone module).

Expects each item to match the structure produced by AnalyzeResponse.model_dump().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Side-by-side comparison derived from full analysis dicts."""

    urls: list[str]
    readability: list[dict]
    sentiment: list[dict]
    common_keywords: list[str]
    unique_keywords_by_url: dict[str, list[str]]
    topic_overlap: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "urls": self.urls,
            "readability": self.readability,
            "sentiment": self.sentiment,
            "common_keywords": self.common_keywords,
            "unique_keywords_by_url": self.unique_keywords_by_url,
            "topic_overlap": self.topic_overlap,
        }


class ContentComparator:
    """Build comparison views from 2–3 completed analysis payloads."""

    def compare(self, analyses: list[dict]) -> ComparisonResult:
        if len(analyses) < 2:
            raise ValueError("At least two analysis results are required")

        urls = [a["metadata"]["url"] for a in analyses]

        readability_rows: list[dict] = []
        sentiment_rows: list[dict] = []

        for a in analyses:
            meta = a["metadata"]
            rd = a["readability"]
            sc = rd["scores"]
            readability_rows.append({
                "url": meta["url"],
                "title": meta.get("title"),
                "word_count": meta.get("word_count"),
                "flesch_reading_ease": sc["flesch_reading_ease"],
                "flesch_kincaid_grade": sc["flesch_kincaid_grade"],
                "gunning_fog": sc["gunning_fog"],
                "smog_index": sc["smog_index"],
                "reading_level": rd["reading_level"],
            })

            overall = a["sentiment"]["overall"]
            sentiment_rows.append({
                "url": meta["url"],
                "label": overall["label"],
                "compound": overall["compound"],
                "distribution": a["sentiment"]["distribution"],
            })

        kw_sets: list[set[str]] = []
        for a in analyses:
            terms = [
                x["keyword"].lower()
                for x in a["keywords"].get("tfidf_keywords", [])
            ]
            kw_sets.append(set(terms))

        common: set[str] = set.intersection(*kw_sets) if kw_sets else set()
        common_list = sorted(common)[:50]

        unique_by_url: dict[str, list[str]] = {}
        for i, a in enumerate(analyses):
            u = a["metadata"]["url"]
            other_union: set[str] = set()
            for j, s in enumerate(kw_sets):
                if j != i:
                    other_union |= s
            unique_sorted = sorted(kw_sets[i] - other_union)[:40]
            unique_by_url[u] = unique_sorted

        topic_sets: list[set[str]] = []
        for a in analyses:
            words: set[str] = set()
            for t in a["topics"].get("lda_topics", []):
                for w in t.get("words", []):
                    words.add(str(w).lower())
            topic_sets.append(words)

        shared = (
            set.intersection(*topic_sets)
            if topic_sets
            else set()
        )
        pairwise: list[dict] = []
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                ta, tb = topic_sets[i], topic_sets[j]
                inter = len(ta & tb)
                union = len(ta | tb) or 1
                pairwise.append({
                    "url_a": urls[i],
                    "url_b": urls[j],
                    "topic_jaccard": round(inter / union, 4),
                    "shared_term_count": inter,
                })

        topic_overlap = {
            "shared_topic_terms": sorted(shared)[:40],
            "pairwise": pairwise,
        }

        return ComparisonResult(
            urls=urls,
            readability=readability_rows,
            sentiment=sentiment_rows,
            common_keywords=common_list,
            unique_keywords_by_url=unique_by_url,
            topic_overlap=topic_overlap,
        )
