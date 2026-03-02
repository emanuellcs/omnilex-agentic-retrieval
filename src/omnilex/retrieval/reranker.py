from __future__ import annotations

import logging
from typing import Any

try:
    from sentence_transformers import CrossEncoder
    import torch
except ImportError:
    CrossEncoder = None
    torch = None

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank candidates using a multilingual Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to use (cuda, cpu). Auto-detects if None.
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size

        if CrossEncoder is None:
            logger.warning(
                "sentence-transformers not installed. CrossEncoderReranker will not work."
            )
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"Loaded cross-encoder model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_field: str = "text",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank a list of candidate documents for a given query.

        Args:
            query: Query string
            candidates: List of candidate document dictionaries
            text_field: Field in candidate dict containing the text to score
            top_k: If specified, return only the top-k results

        Returns:
            Sorted list of candidates with '_reranker_score' added
        """
        if self.model is None:
            raise RuntimeError("Cross-encoder model not loaded.")

        if not candidates:
            return []

        # Prepare pairs: (query, passage)
        # Limit passage length to 512 chars to be safe/efficient
        pairs = [(query, str(c.get(text_field, ""))[:512]) for c in candidates]

        # Get scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Add scores to candidates
        reranked_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = cand.copy()
            cand_copy["_reranker_score"] = float(score)
            reranked_candidates.append(cand_copy)

        # Sort by score descending
        reranked_candidates.sort(key=lambda x: x["_reranker_score"], reverse=True)

        if top_k is not None:
            reranked_candidates = reranked_candidates[:top_k]

        return reranked_candidates

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a list of (query, passage) pairs.

        Args:
            pairs: List of (query, passage) tuples

        Returns:
            List of float scores
        """
        if self.model is None:
            raise RuntimeError("Cross-encoder model not loaded.")

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]
