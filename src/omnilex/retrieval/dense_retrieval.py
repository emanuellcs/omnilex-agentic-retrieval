from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None
    torch = None

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Multilingual text embedder using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | Sequence[str] | None = None,
        batch_size: int = 128,
        chunk_size: int | None = None,
    ):
        """Initialize MultilingualEmbedder.

        Args:
            model_name: HuggingFace model name or local path
            device: Device(s) to use. Auto-detects all available CUDA GPUs if None.
            batch_size: Batch size for encoding
            chunk_size: Number of texts sent to each multi-process worker at a time.
        """
        self.model_name = self._resolve_model_path(model_name)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.pool = None

        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers not installed. MultilingualEmbedder will not work."
            )
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, str):
            self.device = device
        else:
            self.device = list(device)

        model_device = self._get_model_load_device()

        try:
            self.model = SentenceTransformer(self.model_name, device=model_device)
            logger.info(
                "Loaded embedding model %s on %s; encode target device(s): %s",
                self.model_name,
                model_device,
                self._get_encode_devices(),
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            self.model = None

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to a local path if available."""
        # 1. Check if it's already a valid local path
        if Path(model_name).exists():
            return model_name

        # 2. Check Kaggle input directory
        kaggle_path = Path("/kaggle/input/omnilex-models") / model_name.split("/")[-1]
        if kaggle_path.exists():
            return str(kaggle_path)

        # 3. Fallback to original name (HF Hub)
        return model_name

    def _get_model_load_device(self) -> str:
        """Return the device used for the parent process model instance."""
        devices = self._get_encode_devices()

        if len(devices) > 1:
            # Multi-process workers load/use the CUDA devices. Keep the parent
            # model on CPU so it does not reserve memory on cuda:0 before
            # spawning workers.
            return "cpu"

        return devices[0]

    def _get_encode_devices(self) -> list[str]:
        """Return target devices for encoding, using all CUDA GPUs when available."""
        if isinstance(self.device, list):
            return self.device

        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                return [f"cuda:{idx}" for idx in range(device_count)]

        return [self.device]

    def _clear_cuda_cache(self) -> None:
        """Release cached CUDA memory after large encoding jobs."""
        if torch is None or not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug("Failed to clear CUDA cache after encoding: %s", e)

    def start_multi_process_pool(self, target_devices: list[str] | None = None) -> dict:
        """Start a persistent multi-process pool for encoding.

        Args:
            target_devices: List of devices (e.g., ['cuda:0', 'cuda:1']).
                Defaults to all available GPUs.

        Returns:
            The created pool dictionary.
        """
        if self.pool is not None:
            return self.pool

        if target_devices is None:
            target_devices = self._get_encode_devices()

        if len(target_devices) <= 1:
            logger.info(
                "Only one device detected (%s). Skipping pool creation.", target_devices
            )
            return None

        logger.info("Starting persistent multi-process pool on %s", target_devices)
        self.pool = self.model.start_multi_process_pool(target_devices=target_devices)
        return self.pool

    def stop_multi_process_pool(self) -> None:
        """Stop the active multi-process pool and clear CUDA cache."""
        if self.pool is not None:
            logger.info("Stopping multi-process pool...")
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
            self._clear_cuda_cache()

    def _single_process_encode(
        self,
        texts: list[str],
        prefix: str,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode on one device, preferring SentenceTransformers prompt support."""
        try:
            return self.model.encode(
                texts,
                prompt=prefix,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except TypeError:
            prefixed_texts = [prefix + text for text in texts]
            return self.model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

    def _multi_process_encode(
        self,
        texts: list[str],
        prefix: str,
        devices: list[str],
        show_progress: bool,
    ) -> np.ndarray:
        """Encode using SentenceTransformers' native multi-process GPU pool."""
        pool_to_use = self.pool
        own_pool = False

        if pool_to_use is None:
            logger.info(
                "Starting temporary multi-process pool across %s device(s): %s",
                len(devices),
                devices,
            )
            pool_to_use = self.model.start_multi_process_pool(target_devices=devices)
            own_pool = True

        try:
            try:
                return self.model.encode(
                    texts,
                    prompt=prefix,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    pool=pool_to_use,
                    chunk_size=self.chunk_size,
                )
            except TypeError:
                if hasattr(self.model, "encode_multi_process"):
                    try:
                        return self.model.encode_multi_process(
                            texts,
                            pool_to_use,
                            prompt=prefix,
                            batch_size=self.batch_size,
                            chunk_size=self.chunk_size,
                            show_progress_bar=show_progress,
                        )
                    except TypeError:
                        prefixed_texts = [prefix + text for text in texts]
                        return self.model.encode_multi_process(
                            prefixed_texts,
                            pool_to_use,
                            batch_size=self.batch_size,
                            chunk_size=self.chunk_size,
                            show_progress_bar=show_progress,
                        )

                prefixed_texts = [prefix + text for text in texts]
                return self.model.encode(
                    prefixed_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    pool=pool_to_use,
                    chunk_size=self.chunk_size,
                )
        finally:
            if own_pool:
                self.model.stop_multi_process_pool(pool_to_use)
                self._clear_cuda_cache()

    def encode(
        self, texts: list[str], is_query: bool = False, show_progress: bool = True
    ) -> np.ndarray:
        """Encode list of texts into embeddings.

        Args:
            texts: List of text strings
            is_query: Whether these are query texts. Uses 'query: ' for queries and
                'passage: ' for passage texts.
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
        devices = self._get_encode_devices()

        if len(devices) > 1:
            embeddings = self._multi_process_encode(
                texts, prefix, devices, show_progress
            )
        else:
            embeddings = self._single_process_encode(texts, prefix, show_progress)
            self._clear_cuda_cache()

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

    def train(
        self,
        embeddings: np.ndarray,
        index_type: str = "Flat",
        total_expected_docs: int | None = None,
    ) -> None:
        """Initialize and train the FAISS index.

        Args:
            embeddings: Numpy array of embeddings to use for training (N, D)
            index_type: Type of FAISS index ("Flat", "IVFFlat")
            total_expected_docs: Total number of documents expected in the index,
                used to calculate IVFFlat heuristics.
        """
        if faiss is None:
            raise ImportError("faiss is not installed.")

        d = embeddings.shape[1]

        # Create a copy for training to avoid modifying original in-place if not desired
        train_vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(train_vecs)

        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "IVFFlat":
            # Heuristic for nlist
            n_docs = total_expected_docs if total_expected_docs else train_vecs.shape[0]
            nlist = min(100, n_docs // 40)
            if nlist < 1:
                nlist = 1

            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(train_vecs)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

    def add_batch(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        """Add a batch of embeddings and documents to the index.

        Args:
            embeddings: Numpy array of embeddings (N, D)
            documents: List of document dictionaries
        """
        if self.index is None:
            raise RuntimeError(
                "Index must be trained/initialized before adding batches."
            )

        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents.")

        # Normalize for cosine similarity
        embeddings_to_add = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings_to_add)

        self.index.add(embeddings_to_add)
        self.documents.extend(documents)

    def build(
        self, embeddings: np.ndarray, documents: list[dict], index_type: str = "Flat"
    ) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings (N, D)
            documents: List of document dictionaries
            index_type: Type of FAISS index to build ("Flat", "IVFFlat")
        """
        self.train(embeddings, index_type=index_type)
        self.add_batch(embeddings, documents)

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
