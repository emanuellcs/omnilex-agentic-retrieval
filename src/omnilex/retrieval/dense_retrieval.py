from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    faiss = None
    SentenceTransformer = None
    torch = None

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Multilingual text embedder using sentence-transformers (e.g., intfloat/multilingual-e5-large)."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        batch_size: int = 128,
    ):
        """Initialize MultilingualEmbedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda, cpu). Auto-detects if None.
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers not installed. MultilingualEmbedder will not work."
            )
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded embedding model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            self.model = None

    def encode(
        self, texts: list[str], is_query: bool = False, show_progress: bool = True
    ) -> np.ndarray:
        """Encode list of texts into embeddings.

        Args:
            texts: List of text strings
            is_query: Whether these are query texts (prepends 'query: ') or passage texts (prepends 'passage: ')
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings, shape (N, D)
        """
        if self.model is None:
            raise RuntimeError("Embedder model not loaded.")

        if not texts:
            return np.array([])

        # For e5 models, prepend prefix
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + t for t in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: Query string

        Returns:
            Numpy array of embedding, shape (D,)
        """
        embeddings = self.encode([query], is_query=True, show_progress=False)
        return embeddings[0]


class FAISSIndex:
    """FAISS index for efficient dense retrieval."""

    def __init__(
        self, index: faiss.Index | None = None, documents: list[dict] | None = None
    ):
        """Initialize FAISSIndex.

        Args:
            index: Pre-built faiss Index object
            documents: List of document dictionaries
        """
        self.index = index
        self.documents = documents or []

    def build(
        self, embeddings: np.ndarray, documents: list[dict], index_type: str = "Flat"
    ) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings (N, D)
            documents: List of document dictionaries
            index_type: Type of FAISS index to build ("Flat", "IVFFlat")
        """
        if faiss is None:
            raise ImportError("faiss is not installed.")

        self.documents = documents
        d = embeddings.shape[1]

        # Normalize for cosine similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(embeddings)

        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "IVFFlat":
            # Heuristic for nlist
            nlist = min(100, len(documents) // 40)
            if nlist < 1:
                nlist = 1
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        self.index.add(embeddings)

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[dict]:
        """Search the index for nearest neighbors.

        Args:
            query_vector: Single query embedding (D,) or (1, D)
            top_k: Number of results to return

        Returns:
            List of matching documents with '_score' field
        """
        if self.index is None:
            raise RuntimeError("Index not built.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query vector is float32 and normalized
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx].copy()
            doc["_score"] = float(score)
            results.append(doc)

        return results

    def save(self, path: Path | str) -> None:
        """Save FAISS index and documents to disk.

        Args:
            path: Base path to save (saves .faiss and .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, path: Path | str) -> FAISSIndex:
        """Load FAISS index and documents from disk.

        Args:
            path: Base path of saved index

        Returns:
            Loaded FAISSIndex instance
        """
        path = Path(path)

        index = None
        faiss_path = path.with_suffix(".faiss")
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))

        with open(path.with_suffix(".pkl"), "rb") as f:
            documents = pickle.load(f)

        return cls(index=index, documents=documents)
