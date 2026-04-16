"""
Readability Analysis Module

Computes readability metrics for text using the textstat library,
including Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau,
and additional text statistics.
"""

import logging
from dataclasses import dataclass

import textstat

logger = logging.getLogger(__name__)


@dataclass
class ReadabilityResult:
    """Data class holding readability analysis results."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    dale_chall_score: float
    reading_level: str
    estimated_reading_time_minutes: float
    text_statistics: dict

    def to_dict(self) -> dict:
        """Convert readability results to a dictionary."""
        return {
            "scores": {
                "flesch_reading_ease": self.flesch_reading_ease,
                "flesch_kincaid_grade": self.flesch_kincaid_grade,
                "gunning_fog": self.gunning_fog,
                "smog_index": self.smog_index,
                "coleman_liau_index": self.coleman_liau_index,
                "automated_readability_index": self.automated_readability_index,
                "dale_chall_score": self.dale_chall_score,
            },
            "reading_level": self.reading_level,
            "estimated_reading_time_minutes": self.estimated_reading_time_minutes,
            "text_statistics": self.text_statistics,
        }


def _determine_reading_level(flesch_score: float) -> str:
    """
    Map Flesch Reading Ease score to a human-readable difficulty level.

    Args:
        flesch_score: Flesch Reading Ease score (0-100+).

    Returns:
        String description of the reading difficulty level.
    """
    if flesch_score >= 90:
        return "Very Easy (5th grade)"
    elif flesch_score >= 80:
        return "Easy (6th grade)"
    elif flesch_score >= 70:
        return "Fairly Easy (7th grade)"
    elif flesch_score >= 60:
        return "Standard (8th-9th grade)"
    elif flesch_score >= 50:
        return "Fairly Difficult (10th-12th grade)"
    elif flesch_score >= 30:
        return "Difficult (College level)"
    else:
        return "Very Difficult (Graduate level)"


class ReadabilityAnalyzer:
    """
    Analyzes text readability using multiple established metrics.

    Computes Flesch Reading Ease, Flesch-Kincaid Grade Level,
    Gunning Fog Index, SMOG Index, Coleman-Liau Index,
    Automated Readability Index, and Dale-Chall Score.
    Also provides general text statistics and estimated reading time.
    """

    def analyze(self, text: str) -> ReadabilityResult:
        """
        Run full readability analysis on the input text.

        Args:
            text: The text string to analyze.

        Returns:
            ReadabilityResult containing all computed metrics.
        """
        logger.info("Analyzing readability for text of %d characters", len(text))

        # Core readability scores
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        gunning = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        coleman = textstat.coleman_liau_index(text)
        ari = textstat.automated_readability_index(text)
        dale_chall = textstat.dale_chall_readability_score(text)

        # Text statistics
        char_count = textstat.char_count(text, ignore_spaces=True)
        word_count = textstat.lexicon_count(text, removepunct=True)
        sentence_count = textstat.sentence_count(text)
        syllable_count = textstat.syllable_count(text)

        avg_sentence_length = (
            round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        )
        avg_word_length = (
            round(char_count / word_count, 1) if word_count > 0 else 0
        )
        avg_syllables_per_word = (
            round(syllable_count / word_count, 2) if word_count > 0 else 0
        )

        # Estimated reading time (avg 200 words per minute)
        reading_time = round(word_count / 200, 1)

        reading_level = _determine_reading_level(flesch_ease)

        text_stats = {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "syllable_count": syllable_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "avg_syllables_per_word": avg_syllables_per_word,
        }

        return ReadabilityResult(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_grade,
            gunning_fog=gunning,
            smog_index=smog,
            coleman_liau_index=coleman,
            automated_readability_index=ari,
            dale_chall_score=dale_chall,
            reading_level=reading_level,
            estimated_reading_time_minutes=reading_time,
            text_statistics=text_stats,
        )
