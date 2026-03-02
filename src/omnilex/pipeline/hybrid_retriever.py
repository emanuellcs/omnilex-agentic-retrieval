from __future__ import annotations

import logging
from typing import Any

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_retrieval import FAISSIndex, MultilingualEmbedder
from omnilex.retrieval.translator import QueryTranslator
from omnilex.retrieval.citation_graph import CitationCooccurrenceGraph

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates multi-stage retrieval: BM25, Dense, and Citation Graph expansion."""

    def __init__(
        self,
        laws_bm25: BM25Index,
        courts_bm25: BM25Index,
        laws_faiss: FAISSIndex,
        courts_faiss: FAISSIndex,
        embedder: MultilingualEmbedder,
        translator: QueryTranslator,
        citation_graph: CitationCooccurrenceGraph,
        corpus_citation_set: set[str],
    ):
        """Initialize HybridRetriever.

        Args:
            laws_bm25: BM25 index for laws
            courts_bm25: BM25 index for courts
            laws_faiss: FAISS index for laws
            courts_faiss: FAISS index for courts
            embedder: Multilingual embedder
            translator: Query translator (EN -> DE)
            citation_graph: Citation co-occurrence graph
            corpus_citation_set: Set of all valid canonical citation strings
        """
        self.laws_bm25 = laws_bm25
        self.courts_bm25 = courts_bm25
        self.laws_faiss = laws_faiss
        self.courts_faiss = courts_faiss
        self.embedder = embedder
        self.translator = translator
        self.citation_graph = citation_graph
        self.corpus_citation_set = corpus_citation_set

    def retrieve(
        self, query: str, top_k_per_source: int = 50, rrf_k: int = 60
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval and merge results using Reciprocal Rank Fusion (RRF).

        Args:
            query: English query string
            top_k_per_source: Number of candidates to retrieve from each source
            rrf_k: Constant for RRF formula

        Returns:
            Deduplicated list of candidates sorted by RRF score
        """
        # 1. Translate query to German
        german_query = self.translator.translate(query)

        # 2. Run retrieval from all sources
        ranked_lists = []

        # BM25 - English query
        ranked_lists.append(self.laws_bm25.search(query, top_k=top_k_per_source))
        ranked_lists.append(self.courts_bm25.search(query, top_k=top_k_per_source))

        # BM25 - German query
        if german_query != query:
            ranked_lists.append(
                self.laws_bm25.search(german_query, top_k=top_k_per_source)
            )
            ranked_lists.append(
                self.courts_bm25.search(german_query, top_k=top_k_per_source)
            )

        # Dense - English query (multilingual embedder handles it)
        query_emb = self.embedder.encode_query(query)
        ranked_lists.append(self.laws_faiss.search(query_emb, top_k=top_k_per_source))
        ranked_lists.append(self.courts_faiss.search(query_emb, top_k=top_k_per_source))

        # 3. Merge with RRF
        return self.merge_with_rrf(ranked_lists, k=rrf_k)

    def merge_with_rrf(
        self, ranked_lists: list[list[dict[str, Any]]], k: int = 60
    ) -> list[dict[str, Any]]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: List of ranked document lists
            k: RRF constant

        Returns:
            Merged and sorted list of documents
        """
        rrf_scores = {}
        doc_map = {}

        for r_list in ranked_lists:
            for rank, doc in enumerate(r_list):
                citation = doc.get("citation")
                if not citation:
                    continue

                # RRF score: sum(1.0 / (k + rank + 1))
                rrf_scores[citation] = rrf_scores.get(citation, 0.0) + 1.0 / (
                    k + rank + 1
                )

                # Keep the first/best document data (text, etc)
                if citation not in doc_map:
                    doc_map[citation] = doc.copy()

        # Sort citations by RRF score
        sorted_citations = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_results = []
        for citation, score in sorted_citations:
            doc = doc_map[citation]
            doc["_combined_score"] = score
            merged_results.append(doc)

        return merged_results

    def expand_with_graph(
        self,
        candidates: list[dict[str, Any]],
        top_k_seeds: int = 5,
        top_k_expansion: int = 10,
    ) -> list[dict[str, Any]]:
        """Expand candidate pool using citation graph Personalized PageRank.

        Args:
            candidates: Initial list of candidates (sorted by RRF score)
            top_k_seeds: Number of top candidates to use as seeds
            top_k_expansion: Number of new citations to add from graph

        Returns:
            Extended candidate list
        """
        if not candidates:
            return []

        # Get top-k seed citations
        seeds = [c["citation"] for c in candidates[:top_k_seeds]]

        # Run PPR
        ppr_results = self.citation_graph.personalized_pagerank(
            seeds, top_k=top_k_expansion * 2
        )

        # Add new citations not already in candidates
        existing_citations = {c["citation"] for c in candidates}
        expanded_candidates = list(candidates)

        added_count = 0
        for citation, ppr_score in ppr_results:
            if (
                citation not in existing_citations
                and citation in self.corpus_citation_set
            ):
                # Add a dummy doc entry for the graph citation
                # Note: We won't have the text unless we look it up in corpus,
                # but reranker needs text. In the full pipeline, we might need a lookup.
                expanded_candidates.append(
                    {
                        "citation": citation,
                        "text": "",  # Needs to be filled by the caller if reranking is desired
                        "_source": "graph",
                        "_ppr_score": ppr_score,
                    }
                )
                added_count += 1
                if added_count >= top_k_expansion:
                    break

        return expanded_candidates
