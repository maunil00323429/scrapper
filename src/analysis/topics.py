"""
Topic Detection Module

Detects latent topics in text using Latent Dirichlet Allocation (LDA).
"""

import logging
from dataclasses import dataclass

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_NUM_TOPICS: int = 3
DEFAULT_WORDS_PER_TOPIC: int = 8


@dataclass
class TopicResult:
    """Data class holding topic detection results."""

    lda_topics: list[dict]

    def to_dict(self) -> dict:
        """Convert topic results to a dictionary."""
        return {
            "lda_topics": self.lda_topics,
        }


class TopicDetector:
    """
    Detects topics in text using Latent Dirichlet Allocation (LDA).

    Attributes:
        num_topics: Number of topics to extract.
        words_per_topic: Number of representative words shown per topic.
    """

    def __init__(
        self,
        num_topics: int = DEFAULT_NUM_TOPICS,
        words_per_topic: int = DEFAULT_WORDS_PER_TOPIC,
    ) -> None:
        """
        Initialize the topic detector.

        Args:
            num_topics: Number of latent topics to discover.
            words_per_topic: Number of representative words per topic.
        """
        self.num_topics = num_topics
        self.words_per_topic = words_per_topic

    def _format_topics(
        self, model, feature_names: list[str]
    ) -> list[dict]:
        """
        Extract human-readable topics from a fitted topic model.

        Args:
            model: A fitted LDA model.
            feature_names: Vocabulary feature names from the vectorizer.

        Returns:
            List of topic dicts with 'topic_id', 'words', and 'weights'.
        """
        topics = []
        for idx, component in enumerate(model.components_):
            top_indices = component.argsort()[::-1][: self.words_per_topic]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [round(float(component[i]), 4) for i in top_indices]

            topics.append({
                "topic_id": idx + 1,
                "words": top_words,
                "weights": top_weights,
                "label": f"Topic {idx + 1}: {', '.join(top_words[:3])}",
            })

        return topics

    def _detect_lda(self, sentences: list[str]) -> list[dict]:
        """
        Detect topics using Latent Dirichlet Allocation.

        Uses bag-of-words features via CountVectorizer.

        Args:
            sentences: List of sentence strings.

        Returns:
            List of topic dicts from LDA analysis.
        """
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )

        try:
            doc_term_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            logger.warning("LDA vectorization failed")
            return []

        # Adjust topics if fewer sentences than requested topics
        actual_topics = min(self.num_topics, len(sentences))
        if actual_topics < 2:
            return []

        lda_model = LatentDirichletAllocation(
            n_components=actual_topics,
            max_iter=20,
            learning_method="online",
            random_state=42,
        )
        lda_model.fit(doc_term_matrix)

        feature_names = vectorizer.get_feature_names_out().tolist()
        return self._format_topics(lda_model, feature_names)

    def detect(self, sentences: list[str]) -> TopicResult:
        """
        Run LDA topic detection on the input sentences.

        Args:
            sentences: List of sentence strings from the text.

        Returns:
            TopicResult containing topics from LDA.
        """
        logger.info(
            "Detecting %d topics from %d sentences",
            self.num_topics,
            len(sentences),
        )

        lda_topics = self._detect_lda(sentences)

        return TopicResult(
            lda_topics=lda_topics,
        )
