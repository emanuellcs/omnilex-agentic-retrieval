import numpy as np
import unittest
from unittest.mock import MagicMock, patch

from omnilex.retrieval import reranker
from omnilex.retrieval.reranker import CrossEncoderReranker

mock_st = MagicMock()
mock_torch = MagicMock()


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        self.cross_encoder_patch = patch.object(
            reranker,
            "CrossEncoder",
            mock_st.CrossEncoder,
        )
        self.torch_patch = patch.object(reranker, "torch", mock_torch)
        self.cross_encoder_patch.start()
        self.torch_patch.start()
        mock_st.CrossEncoder.reset_mock()
        mock_torch.cuda.is_available.return_value = False
        self.mock_model = MagicMock()
        mock_st.CrossEncoder.return_value = self.mock_model

    def tearDown(self):
        self.torch_patch.stop()
        self.cross_encoder_patch.stop()

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
