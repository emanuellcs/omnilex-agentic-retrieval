import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import networkx as nx

from omnilex.retrieval.citation_graph import CitationCooccurrenceGraph


class TestCitationCooccurrenceGraph(unittest.TestCase):
    def test_extract_citations(self):
        graph = CitationCooccurrenceGraph()
        text = "According to BGE 116 Ia 56 E. 2 and Art. 11 Abs. 2 OR, also 5A_800/2019 E. 3.1 is relevant."

        citations = graph.extract_citations_from_text(text)

        self.assertIn("BGE 116 Ia 56 E. 2", citations)
        self.assertIn("Art. 11 Abs. 2 OR", citations)
        self.assertIn("5A_800/2019 E. 3.1", citations)

    def test_build_graph(self):
        df = pd.DataFrame(
            {
                "citation": ["BGE 100 I 1", "BGE 100 I 10"],
                "text": ["See Art. 1 ZGB and BGE 100 I 10.", "Mentions Art. 1 ZGB."],
            }
        )

        graph = CitationCooccurrenceGraph()
        graph.build_from_corpus(df)

        # Nodes should exist
        self.assertTrue(graph.graph.has_node("BGE 100 I 1"))
        self.assertTrue(graph.graph.has_node("BGE 100 I 10"))
        self.assertTrue(graph.graph.has_node("Art. 1 ZGB"))

        # Edge weight should be correct
        # Art. 1 ZGB appears in both docs
        # In doc 1, it's connected to BGE 100 I 1 and BGE 100 I 10
        # In doc 2, it's connected to BGE 100 I 10
        self.assertEqual(graph.graph["Art. 1 ZGB"]["BGE 100 I 10"]["weight"], 2.0)

    def test_pagerank(self):
        graph = CitationCooccurrenceGraph()
        # Create a simple triangle
        graph.graph.add_edge("A", "B", weight=1.0)
        graph.graph.add_edge("B", "C", weight=1.0)
        graph.graph.add_edge("C", "A", weight=1.0)

        results = graph.personalized_pagerank(["A"], top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "A")  # Seed usually gets highest score

    def test_save_load(self):
        graph = CitationCooccurrenceGraph()
        graph.graph.add_edge("A", "B", weight=5.0)
        graph.citation_to_corpus_freq["A"] = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph.pkl"
            graph.save(save_path)

            loaded = CitationCooccurrenceGraph.load(save_path)
            self.assertEqual(loaded.graph["A"]["B"]["weight"], 5.0)
            self.assertEqual(loaded.citation_to_corpus_freq["A"], 10)


if __name__ == "__main__":
    unittest.main()
