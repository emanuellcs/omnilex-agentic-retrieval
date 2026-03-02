from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from omnilex.pipeline.hybrid_retriever import HybridRetriever
from omnilex.retrieval.reranker import CrossEncoderReranker
from omnilex.evaluation.metrics import macro_f1

logger = logging.getLogger(__name__)


class FullPipeline:
    """End-to-end legal citation retrieval pipeline."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        threshold: float = 0.0,
        corpus_citation_set: set[str] | None = None,
    ):
        """Initialize FullPipeline.

        Args:
            hybrid_retriever: Orchestrator for retrieval and expansion
            reranker: Cross-encoder for second-stage ranking
            threshold: Score threshold for final selection
            corpus_citation_set: Valid canonical citations for grounding
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.threshold = threshold
        self.corpus_citation_set = (
            corpus_citation_set or hybrid_retriever.corpus_citation_set
        )

    def run_query(self, query: str) -> list[str]:
        """Process a single query through the full pipeline.

        Args:
            query: English legal query

        Returns:
            List of selected canonical citation strings
        """
        # Stage 1: Hybrid Retrieval
        candidates = self.hybrid_retriever.retrieve(query)

        # Stage 2: Graph Expansion
        expanded_candidates = self.hybrid_retriever.expand_with_graph(candidates)

        # NOTE: Graph candidates might lack 'text' field.
        # In a real scenario, we'd look it up. For now, reranker will handle "" text.

        # Stage 3: Cross-Encoder Reranking
        reranked = self.reranker.rerank(query, expanded_candidates)

        # Stage 4: Thresholding and Grounding
        selected = []
        for cand in reranked:
            if cand["_reranker_score"] >= self.threshold:
                cit = cand["citation"]
                if cit in self.corpus_citation_set:
                    selected.append(cit)

        return list(set(selected))

    def run_batch(
        self, queries: list[str], show_progress: bool = True
    ) -> list[list[str]]:
        """Process a batch of queries.

        Args:
            queries: List of English queries
            show_progress: Whether to show progress bar

        Returns:
            List of citation lists (one per query)
        """
        results = []
        iterator = (
            tqdm(queries, desc="Pipeline inference") if show_progress else queries
        )
        for q in iterator:
            results.append(self.run_query(q))
        return results

    def tune_threshold(
        self,
        val_queries: list[str],
        val_gold: list[list[str]],
        threshold_range: list[float] | None = None,
    ) -> float:
        """Find the optimal threshold on a validation set to maximize macro F1.

        Args:
            val_queries: List of validation queries
            val_gold: List of gold citation sets
            threshold_range: List of threshold values to test

        Returns:
            Best threshold found
        """
        if threshold_range is None:
            threshold_range = list(np.arange(-5.0, 5.01, 0.25))

        logger.info(f"Tuning threshold over {len(threshold_range)} values...")

        # Pre-calculate reranker scores for all candidates of all val queries
        all_val_candidates_with_scores = []
        for q in tqdm(val_queries, desc="Pre-scoring val queries"):
            candidates = self.hybrid_retriever.retrieve(q)
            expanded = self.hybrid_retriever.expand_with_graph(candidates)
            reranked = self.reranker.rerank(q, expanded)
            all_val_candidates_with_scores.append(reranked)

        best_f1 = -1.0
        best_threshold = self.threshold

        for t in threshold_range:
            predictions = []
            for candidates in all_val_candidates_with_scores:
                preds = [c["citation"] for c in candidates if c["_reranker_score"] >= t]
                # Apply grounding
                preds = [p for p in preds if p in self.corpus_citation_set]
                predictions.append(list(set(preds)))

            metrics = macro_f1(predictions, val_gold)
            f1 = metrics["macro_f1"]

            logger.info(f"Threshold: {t:.2f} -> Macro F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        logger.info(
            f"Best threshold: {best_threshold:.2f} with Macro F1: {best_f1:.4f}"
        )
        self.threshold = float(best_threshold)
        return self.threshold

    def save_config(self, path: Path | str) -> None:
        """Save pipeline configuration to JSON.

        Args:
            path: Output JSON path
        """
        config = {
            "threshold": self.threshold,
            "embedder_model": getattr(self.hybrid_retriever.embedder, "model_name", ""),
            "reranker_model": getattr(self.reranker, "model_name", ""),
            "rrf_k": 60,  # Default
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: Path | str) -> None:
        """Load pipeline configuration from JSON.

        Args:
            path: Input JSON path
        """
        with open(path, "r") as f:
            config = json.load(f)
        self.threshold = config.get("threshold", self.threshold)
