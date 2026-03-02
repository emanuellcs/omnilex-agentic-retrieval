from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd
from tqdm import tqdm

from omnilex.citations.normalizer import CitationNormalizer

logger = logging.getLogger(__name__)


class CitationCooccurrenceGraph:
    """Graph where nodes are citations and edges represent co-occurrence in legal texts."""

    # Docket-style pattern: e.g., 5A_800/2019 E. 2
    DOCKET_PATTERN = r"\b\d[A-Z]_\d+/\d{4}\s+(?:E\.?|cons\.?|Erw\.?)\s*[\d.]+\b"

    def __init__(self):
        """Initialize CitationCooccurrenceGraph."""
        self.graph = nx.Graph()
        self.citation_to_corpus_freq = defaultdict(int)
        self.normalizer = CitationNormalizer()

    def extract_citations_from_text(self, text: str) -> list[str]:
        """Extract all valid citation strings from text.

        Args:
            text: Input text string

        Returns:
            List of canonical citation strings
        """
        if not text:
            return []

        found_citations = []

        # 1. Use CitationNormalizer patterns (BGE and Law Abbrevs)
        # Note: Normalizer.normalize() is designed for single citations,
        # but we can use its regexes for extraction.

        # Extract BGEs
        for match in re.finditer(self.normalizer.BGE_PATTERN, text, re.IGNORECASE):
            raw = match.group(0)
            canonical = self.normalizer.canonicalize(raw)
            if canonical:
                found_citations.append(canonical)

        # Extract Law citations (requires knowing abbreviations)
        for abbrev in self.normalizer._law_abbreviations:
            # Simple check for abbreviation presence
            if abbrev in text:
                # Find occurrences around the abbreviation
                # This is a bit heuristic but avoids re-implementing complex regex
                pattern = rf"(?:Art\.?|Artikel)\s*\d+[a-z]?.*?\s+{re.escape(abbrev)}"
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    canonical = self.normalizer.canonicalize(match.group(0))
                    if canonical:
                        found_citations.append(canonical)

        # 2. Extract Docket-style citations
        for match in re.finditer(self.DOCKET_PATTERN, text, re.IGNORECASE):
            # These are already quite canonical in form
            found_citations.append(match.group(0).strip())

        return list(set(found_citations))

    def build_from_corpus(
        self,
        corpus_df: pd.DataFrame,
        text_field: str = "text",
        citation_field: str = "citation",
        max_rows: int | None = None,
    ) -> None:
        """Build the graph from a corpus dataframe.

        Args:
            corpus_df: Dataframe with citations and text
            text_field: Column name for text
            citation_field: Column name for the document's own citation
            max_rows: Limit number of rows to process
        """
        if max_rows:
            corpus_df = corpus_df.head(max_rows)

        logger.info(f"Building citation graph from {len(corpus_df)} rows...")

        for _, row in tqdm(
            corpus_df.iterrows(), total=len(corpus_df), desc="Processing corpus"
        ):
            doc_citation = row[citation_field]
            text = str(row[text_field])

            if pd.isna(doc_citation):
                continue

            # Update corpus frequency
            self.citation_to_corpus_freq[doc_citation] += 1

            # Extract citations mentioned in text
            mentions = self.extract_citations_from_text(text)

            # Add node for doc_citation
            if not self.graph.has_node(doc_citation):
                self.graph.add_node(doc_citation)

            # Connect mentions to doc_citation and to each other
            all_cits = list(set(mentions + [doc_citation]))

            for i in range(len(all_cits)):
                cit_i = all_cits[i]
                if not self.graph.has_node(cit_i):
                    self.graph.add_node(cit_i)

                for j in range(i + 1, len(all_cits)):
                    cit_j = all_cits[j]
                    if not self.graph.has_node(cit_j):
                        self.graph.add_node(cit_j)

                    # Increment edge weight
                    if self.graph.has_edge(cit_i, cit_j):
                        self.graph[cit_i][cit_j]["weight"] += 1.0
                    else:
                        self.graph.add_edge(cit_i, cit_j, weight=1.0)

    def get_neighbors(self, citation: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Get top-k neighbors of a citation based on edge weight.

        Args:
            citation: Source citation string
            top_k: Number of neighbors to return

        Returns:
            List of (citation, weight) tuples
        """
        if not self.graph.has_node(citation):
            return []

        neighbors = []
        for n in self.graph.neighbors(citation):
            weight = self.graph[citation][n]["weight"]
            neighbors.append((n, weight))

        # Sort by weight descending
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]

    def personalized_pagerank(
        self, seed_citations: list[str], top_k: int = 20, damping: float = 0.85
    ) -> list[tuple[str, float]]:
        """Compute Personalized PageRank from seed nodes.

        Args:
            seed_citations: List of citations to seed the random walk
            top_k: Number of results to return
            damping: Damping factor for PageRank

        Returns:
            List of (citation, score) tuples
        """
        # Filter seeds to those in graph
        valid_seeds = [s for s in seed_citations if self.graph.has_node(s)]
        if not valid_seeds:
            return []

        # Create personalization dict
        personalization = {node: 0.0 for node in self.graph.nodes()}
        weight = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            personalization[s] = weight

        try:
            # Use weight attribute for edges
            scores = nx.pagerank(
                self.graph,
                alpha=damping,
                personalization=personalization,
                weight="weight",
            )

            # Sort scores
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Exclude seeds from results if desired, or keep them
            return sorted_scores[:top_k]
        except Exception as e:
            logger.error(f"PageRank computation failed: {e}")
            return []

    def save(self, path: Path | str) -> None:
        """Save graph structure to disk.

        Args:
            path: Path to save pickle file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"graph": self.graph, "freq": dict(self.citation_to_corpus_freq)}

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> CitationCooccurrenceGraph:
        """Load graph structure from disk.

        Args:
            path: Path to pickle file

        Returns:
            Loaded CitationCooccurrenceGraph instance
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.graph = data["graph"]
        instance.citation_to_corpus_freq = defaultdict(int, data["freq"])
        return instance
