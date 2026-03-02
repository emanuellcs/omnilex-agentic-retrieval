import unittest
from unittest.mock import MagicMock, patch
import sys

# We need to mock these BEFORE the class definition if we import at module level,
# but it's better to import INSIDE tests or use a localized patch.


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        # Mocking the dependencies for the module imports inside the class
        self.mocks = {
            "faiss": MagicMock(),
            "sentence_transformers": MagicMock(),
            "torch": MagicMock(),
            "transformers": MagicMock(),
            "networkx": MagicMock(),
        }
        self.mocks["torch"].__spec__ = MagicMock()

        with patch.dict(sys.modules, self.mocks):
            from omnilex.pipeline.hybrid_retriever import HybridRetriever

            self.mock_laws_bm25 = MagicMock()
            self.mock_courts_bm25 = MagicMock()
            self.mock_laws_faiss = MagicMock()
            self.mock_courts_faiss = MagicMock()
            self.mock_embedder = MagicMock()
            self.mock_translator = MagicMock()
            self.mock_graph = MagicMock()
            self.corpus_citation_set = {"A", "B", "C", "D", "E"}

            self.retriever = HybridRetriever(
                self.mock_laws_bm25,
                self.mock_courts_bm25,
                self.mock_laws_faiss,
                self.mock_courts_faiss,
                self.mock_embedder,
                self.mock_translator,
                self.mock_graph,
                self.corpus_citation_set,
            )

    def test_rrf_merge(self):
        # Two lists with some overlap
        list1 = [{"citation": "A", "score": 10}, {"citation": "B", "score": 9}]
        list2 = [{"citation": "B", "score": 8}, {"citation": "C", "score": 7}]

        merged = self.retriever.merge_with_rrf([list1, list2], k=60)

        self.assertEqual(merged[0]["citation"], "B")
        self.assertEqual(merged[1]["citation"], "A")
        self.assertEqual(merged[2]["citation"], "C")

    def test_expand_with_graph(self):
        candidates = [{"citation": "A"}, {"citation": "B"}]
        self.mock_graph.personalized_pagerank.return_value = [
            ("A", 0.5),
            ("B", 0.3),
            ("C", 0.1),
            ("D", 0.05),
        ]

        expanded = self.retriever.expand_with_graph(
            candidates, top_k_seeds=2, top_k_expansion=1
        )

        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[2]["citation"], "C")
        self.assertEqual(expanded[2]["_source"], "graph")


if __name__ == "__main__":
    unittest.main()
