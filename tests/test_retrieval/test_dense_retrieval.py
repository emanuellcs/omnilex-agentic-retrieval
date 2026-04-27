import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Mock sentence-transformers, faiss, and torch before importing
mock_st = MagicMock()
mock_faiss = MagicMock()
mock_torch = MagicMock()

with patch.dict(
    sys.modules,
    {"sentence_transformers": mock_st, "faiss": mock_faiss, "torch": mock_torch},
):
    from omnilex.retrieval.dense_retrieval import FAISSIndex, MultilingualEmbedder


class TestMultilingualEmbedder(unittest.TestCase):
    def setUp(self):
        mock_st.SentenceTransformer.reset_mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        self.mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = self.mock_model

    def test_init(self):
        embedder = MultilingualEmbedder()
        mock_st.SentenceTransformer.assert_called_once()
        self.assertIsNotNone(embedder.model)

    def test_encode(self):
        embedder = MultilingualEmbedder()
        self.mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        embeddings = embedder.encode(["test1", "test2"])

        self.assertEqual(embeddings.shape, (2, 2))
        self.mock_model.encode.assert_called_once()
        args, kwargs = self.mock_model.encode.call_args
        self.assertEqual(args[0], ["test1", "test2"])
        self.assertEqual(kwargs["prompt"], "passage: ")

    def test_encode_query(self):
        embedder = MultilingualEmbedder()
        self.mock_model.encode.return_value = np.array([[0.1, 0.2]])

        embedding = embedder.encode_query("query text")

        self.assertEqual(embedding.shape, (2,))
        args, kwargs = self.mock_model.encode.call_args
        self.assertEqual(args[0], ["query text"])
        self.assertEqual(kwargs["prompt"], "query: ")

    def test_encode_uses_multi_gpu_pool(self):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        self.mock_model.start_multi_process_pool.return_value = {
            "processes": ["p0", "p1"]
        }
        self.mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        embedder = MultilingualEmbedder(batch_size=1, chunk_size=1)
        embeddings = embedder.encode(["test1", "test2"])

        mock_st.SentenceTransformer.assert_called_once_with(
            embedder.model_name,
            device="cpu",
        )
        self.assertEqual(embeddings.shape, (2, 2))
        self.mock_model.start_multi_process_pool.assert_called_once_with(
            target_devices=["cuda:0", "cuda:1"]
        )
        self.mock_model.stop_multi_process_pool.assert_called_once_with(
            {"processes": ["p0", "p1"]}
        )
        args, kwargs = self.mock_model.encode.call_args
        self.assertEqual(args[0], ["test1", "test2"])
        self.assertEqual(kwargs["prompt"], "passage: ")
        self.assertEqual(kwargs["pool"], {"processes": ["p0", "p1"]})
        self.assertEqual(kwargs["chunk_size"], 1)


class TestFAISSIndex(unittest.TestCase):
    def setUp(self):
        mock_faiss.reset_mock()

    def test_build_and_search(self):
        # We need a real-ish faiss for some parts or just mock everything
        # Let's mock the index object
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))

        docs = [
            {"citation": "A", "text": "text A"},
            {"citation": "B", "text": "text B"},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        idx = FAISSIndex()
        idx.build(embeddings, docs)

        self.assertEqual(len(idx.documents), 2)
        mock_faiss.IndexFlatIP.assert_called_once()

        results = idx.search(np.array([1.0, 0.0]))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["citation"], "A")
        self.assertEqual(results[0]["_score"], 0.9)

    def test_save_load(self):
        # Mock index
        mock_index = MagicMock()
        docs = [{"citation": "A"}]

        idx = FAISSIndex(index=mock_index, documents=docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"

            idx.save(save_path)

            self.assertTrue(save_path.with_suffix(".pkl").exists())
            mock_faiss.write_index.assert_called_once()

            # Load
            mock_faiss.read_index.return_value = mock_index
            loaded_idx = FAISSIndex.load(save_path)

            self.assertEqual(len(loaded_idx.documents), 1)
            self.assertEqual(loaded_idx.documents[0]["citation"], "A")


if __name__ == "__main__":
    unittest.main()
