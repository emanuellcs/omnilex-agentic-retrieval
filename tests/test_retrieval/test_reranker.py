import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Mock sentence-transformers and torch before importing
mock_st = MagicMock()
mock_torch = MagicMock()

with patch.dict(sys.modules, {"sentence_transformers": mock_st, "torch": mock_torch}):
    from omnilex.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        mock_st.CrossEncoder.reset_mock()
        self.mock_model = MagicMock()
        mock_st.CrossEncoder.return_value = self.mock_model

    def test_init(self):
        reranker = CrossEncoderReranker()
        mock_st.CrossEncoder.assert_called_once()
        self.assertIsNotNone(reranker.model)

    def test_rerank_sorts_by_score(self):
        reranker = CrossEncoderReranker()
        # Mock scores: first candidate gets low score, second gets high score
        self.mock_model.predict.return_value = np.array([0.1, 0.9])

        candidates = [
            {"citation": "A", "text": "text A"},
            {"citation": "B", "text": "text B"},
        ]

        results = reranker.rerank("query text", candidates)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["citation"], "B")  # B has higher score (0.9)
        self.assertEqual(results[0]["_reranker_score"], 0.9)
        self.assertEqual(results[1]["citation"], "A")
        self.assertEqual(results[1]["_reranker_score"], 0.1)

    def test_rerank_top_k(self):
        reranker = CrossEncoderReranker()
        self.mock_model.predict.return_value = np.array([0.5, 0.9, 0.1])

        candidates = [
            {"citation": "A", "text": "text A"},
            {"citation": "B", "text": "text B"},
            {"citation": "C", "text": "text C"},
        ]

        results = reranker.rerank("query text", candidates, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["citation"], "B")
        self.assertEqual(results[1]["citation"], "A")

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        self.assertEqual(reranker.rerank("query text", []), [])


if __name__ == "__main__":
    unittest.main()
