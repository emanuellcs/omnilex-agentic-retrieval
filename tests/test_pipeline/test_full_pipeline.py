import unittest
from unittest.mock import MagicMock, patch
import sys


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.mocks = {
            "faiss": MagicMock(),
            "sentence_transformers": MagicMock(),
            "torch": MagicMock(),
            "transformers": MagicMock(),
            "networkx": MagicMock(),
        }
        self.mocks["torch"].__spec__ = MagicMock()

        with patch.dict(sys.modules, self.mocks):
            from omnilex.pipeline.full_pipeline import FullPipeline

            self.mock_hybrid = MagicMock()
            self.mock_reranker = MagicMock()
            self.corpus_set = {"A", "B", "C"}
            self.pipeline = FullPipeline(
                self.mock_hybrid,
                self.mock_reranker,
                threshold=0.5,
                corpus_citation_set=self.corpus_set,
            )

    def test_run_query(self):
        # Mock hybrid retrieval
        self.mock_hybrid.retrieve.return_value = [{"citation": "A"}]
        # Mock expansion
        self.mock_hybrid.expand_with_graph.return_value = [
            {"citation": "A"},
            {"citation": "B"},
        ]
        # Mock reranker scores
        self.mock_reranker.rerank.return_value = [
            {"citation": "B", "_reranker_score": 0.9},
            {"citation": "A", "_reranker_score": 0.1},
        ]

        results = self.pipeline.run_query("query")

        # Only B should be selected because score 0.9 >= threshold 0.5
        # A score 0.1 < 0.5
        self.assertEqual(results, ["B"])

    def test_hard_grounding(self):
        # Score > threshold but citation not in corpus_set
        self.mock_hybrid.retrieve.return_value = []
        self.mock_hybrid.expand_with_graph.return_value = []
        self.mock_reranker.rerank.return_value = [
            {"citation": "INVALID", "_reranker_score": 1.0}
        ]

        results = self.pipeline.run_query("query")
        self.assertEqual(results, [])

    def test_tune_threshold(self):
        # Simplified tuning test
        val_queries = ["q1"]
        val_gold = [["A"]]

        self.mock_hybrid.retrieve.return_value = []
        self.mock_hybrid.expand_with_graph.return_value = []
        self.mock_reranker.rerank.return_value = [
            {"citation": "A", "_reranker_score": 0.6},
            {"citation": "B", "_reranker_score": 0.4},
        ]

        # Test a couple of thresholds
        # If threshold is 0.5, prediction is ["A"] -> F1=1.0
        # If threshold is 0.7, prediction is [] -> F1=0.0
        best_t = self.pipeline.tune_threshold(
            val_queries, val_gold, threshold_range=[0.5, 0.7]
        )

        self.assertEqual(best_t, 0.5)
        self.assertEqual(self.pipeline.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
