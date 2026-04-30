from __future__ import annotations

import logging
import multiprocessing as py_mp
import os
import pickle
import queue
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _cuda_device_index(device: str) -> int:
    if ":" not in str(device):
        return 0
    return int(str(device).split(":", 1)[1])


def _normalize_dtype(dtype: str | None, device: str) -> str:
    value = (dtype or "auto").lower()
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
        "auto": "auto",
    }
    value = aliases.get(value, value)
    if value == "auto":
        return "float16" if _is_cuda_device(device) else "float32"
    if value not in {"float16", "float32"}:
        raise ValueError("dtype must be one of: auto, float16, fp16, float32, fp32")
    return value


def _torch_dtype_for_device(dtype: str | None, device: str, torch_module: Any) -> Any:
    normalized = _normalize_dtype(dtype, device)
    if normalized == "float16" and _is_cuda_device(device):
        return getattr(torch_module, "float16", None)
    if normalized == "float32":
        return getattr(torch_module, "float32", None)
    return None


def _load_sentence_transformer(
    model_name: str,
    device: str,
    dtype: str | None,
    max_seq_length: int | None,
):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed.")

    model_kwargs = {}
    torch_dtype = (
        _torch_dtype_for_device(dtype, device, torch) if torch is not None else None
    )
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    try:
        if model_kwargs:
            model = SentenceTransformer(
                model_name, device=device, model_kwargs=model_kwargs
            )
        else:
            model = SentenceTransformer(model_name, device=device)
    except TypeError:
        model = SentenceTransformer(model_name, device=device)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    if (
        torch is not None
        and _normalize_dtype(dtype, device) == "float16"
        and _is_cuda_device(device)
        and hasattr(model, "half")
    ):
        model.half()

    return model


def _encode_with_prompt(
    model: Any,
    texts: list[str],
    prefix: str,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    try:
        embeddings = model.encode(
            texts,
            prompt=prefix,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    except TypeError:
        prefixed_texts = [prefix + text for text in texts]
        embeddings = model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    return np.asarray(embeddings, dtype=np.float32)


def _embedding_worker_main(
    worker_id: int,
    model_name: str,
    device: str,
    batch_size: int,
    dtype: str,
    max_seq_length: int | None,
    task_queue,
    result_queue,
) -> None:
    pid = os.getpid()
    try:
        import torch as worker_torch
        from sentence_transformers import (
            SentenceTransformer as WorkerSentenceTransformer,
        )

        if _is_cuda_device(device):
            worker_torch.cuda.set_device(_cuda_device_index(device))

        worker_dtype = _normalize_dtype(dtype, device)
        model_kwargs = {}
        torch_dtype = _torch_dtype_for_device(worker_dtype, device, worker_torch)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        try:
            if model_kwargs:
                model = WorkerSentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            else:
                model = WorkerSentenceTransformer(model_name, device=device)
        except TypeError:
            model = WorkerSentenceTransformer(model_name, device=device)

        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        if (
            worker_dtype == "float16"
            and _is_cuda_device(device)
            and hasattr(model, "half")
        ):
            model.half()
        if hasattr(model, "eval"):
            model.eval()

        _encode_with_prompt(
            model,
            ["warmup"],
            "passage: ",
            batch_size=1,
            show_progress=False,
        )

        memory_allocated = None
        if _is_cuda_device(device):
            worker_torch.cuda.synchronize(_cuda_device_index(device))
            memory_allocated = worker_torch.cuda.memory_allocated(
                _cuda_device_index(device)
            )

        result_queue.put(
            {
                "type": "ready",
                "worker_id": worker_id,
                "pid": pid,
                "device": device,
                "memory_allocated": memory_allocated,
            }
        )
    except Exception:
        result_queue.put(
            {
                "type": "error",
                "phase": "init",
                "worker_id": worker_id,
                "pid": pid,
                "device": device,
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            embeddings = _encode_with_prompt(
                model,
                task["texts"],
                task["prefix"],
                batch_size=task["batch_size"],
                show_progress=False,
            )
            result_queue.put(
                {
                    "type": "result",
                    "task_id": task["task_id"],
                    "chunk_index": task["chunk_index"],
                    "embeddings": embeddings,
                    "worker_id": worker_id,
                    "pid": pid,
                    "device": device,
                }
            )
        except Exception:
            result_queue.put(
                {
                    "type": "error",
                    "phase": "encode",
                    "task_id": task.get("task_id"),
                    "chunk_index": task.get("chunk_index"),
                    "worker_id": worker_id,
                    "pid": pid,
                    "device": device,
                    "traceback": traceback.format_exc(),
                }
            )


class EmbeddingWorkerPool:
    """Spawned embedding workers with explicit one-process-per-device ownership."""

    def __init__(
        self,
        model_name: str,
        devices: list[str],
        batch_size: int,
        chunk_size: int | None = None,
        dtype: str = "float16",
        max_seq_length: int | None = None,
        startup_timeout: float = 180.0,
    ):
        self.model_name = model_name
        self.devices = devices
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.startup_timeout = startup_timeout
        self._ctx = py_mp.get_context("spawn")
        self.input_queue = self._ctx.Queue(maxsize=max(len(devices) * 2, 1))
        self.output_queue = self._ctx.Queue()
        self.processes = []
        self.ready_workers: dict[int, dict[str, Any]] = {}
        self._next_task_id = 0
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return

        for worker_id, device in enumerate(self.devices):
            process = self._ctx.Process(
                target=_embedding_worker_main,
                args=(
                    worker_id,
                    self.model_name,
                    device,
                    self.batch_size,
                    self.dtype,
                    self.max_seq_length,
                    self.input_queue,
                    self.output_queue,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

        self._wait_until_ready()
        self._started = True

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        ready_count = 0
        while ready_count < len(self.devices):
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(
                    "Timed out waiting for embedding workers to initialize "
                    f"after {self.startup_timeout:.0f}s."
                )
            self._raise_if_worker_died()
            try:
                message = self.output_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if message.get("type") == "ready":
                ready_count += 1
                self.ready_workers[message["worker_id"]] = message
                memory_gb = (
                    message["memory_allocated"] / 1024**3
                    if message.get("memory_allocated") is not None
                    else None
                )
                if memory_gb is None:
                    logger.info(
                        "Embedding worker %s ready on %s (pid=%s)",
                        message["worker_id"],
                        message["device"],
                        message["pid"],
                    )
                else:
                    logger.info(
                        "Embedding worker %s ready on %s (pid=%s, torch_allocated=%.2f GB)",
                        message["worker_id"],
                        message["device"],
                        message["pid"],
                        memory_gb,
                    )
            elif message.get("type") == "error":
                self.stop()
                raise RuntimeError(self._format_worker_error(message))

    def _raise_if_worker_died(self) -> None:
        for process in self.processes:
            if process.exitcode is not None and process.exitcode != 0:
                self.stop()
                raise RuntimeError(
                    f"Embedding worker pid={process.pid} exited unexpectedly "
                    f"with code {process.exitcode}."
                )

    def _format_worker_error(self, message: dict[str, Any]) -> str:
        return (
            f"Embedding worker failed during {message.get('phase')} "
            f"on {message.get('device')} (worker_id={message.get('worker_id')}, "
            f"pid={message.get('pid')}):\n{message.get('traceback')}"
        )

    def _default_chunk_size(self, text_count: int) -> int:
        if self.chunk_size is not None:
            return max(int(self.chunk_size), 1)
        chunk_size = min(
            max(int(np.ceil(text_count / max(len(self.processes), 1) / 10)), 1), 5000
        )
        return chunk_size

    def encode(
        self,
        texts: list[str],
        prefix: str,
        show_progress: bool = True,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        if not self._started:
            self.start()
        if self._closed:
            raise RuntimeError("Embedding worker pool is closed.")
        if not texts:
            return np.array([], dtype=np.float32)

        effective_chunk_size = chunk_size or self._default_chunk_size(len(texts))
        starts = list(range(0, len(texts), effective_chunk_size))
        results: dict[int, np.ndarray] = {}
        next_chunk_index = 0
        pending_task_ids: set[int] = set()
        max_in_flight = max(len(self.processes) * 2, 1)

        def submit_next() -> bool:
            nonlocal next_chunk_index
            if next_chunk_index >= len(starts):
                return False
            start = starts[next_chunk_index]
            task_id = self._next_task_id
            self._next_task_id += 1
            self.input_queue.put(
                {
                    "task_id": task_id,
                    "chunk_index": next_chunk_index,
                    "texts": texts[start : start + effective_chunk_size],
                    "prefix": prefix,
                    "batch_size": self.batch_size,
                }
            )
            pending_task_ids.add(task_id)
            next_chunk_index += 1
            return True

        while len(pending_task_ids) < max_in_flight and submit_next():
            pass

        while pending_task_ids:
            self._raise_if_worker_died()
            try:
                message = self.output_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if message.get("type") == "result":
                pending_task_ids.discard(message["task_id"])
                results[message["chunk_index"]] = message["embeddings"]
                while len(pending_task_ids) < max_in_flight and submit_next():
                    pass
            elif message.get("type") == "error":
                raise RuntimeError(self._format_worker_error(message))

        embeddings = [results[idx] for idx in range(len(starts))]
        return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True

        for _ in self.processes:
            try:
                self.input_queue.put_nowait(None)
            except Exception:
                pass

        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


class MultilingualEmbedder:
    """Multilingual text embedder using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | Sequence[str] | None = None,
        devices: Sequence[str] | None = None,
        batch_size: int = 128,
        chunk_size: int | None = None,
        dtype: str = "float16",
        max_seq_length: int | None = None,
    ):
        """Initialize MultilingualEmbedder.

        Args:
            model_name: HuggingFace model name or local path
            device: Device(s) to use. Auto-detects all available CUDA GPUs if None.
            devices: Explicit list of devices. Takes precedence over device.
            batch_size: Batch size for encoding
            chunk_size: Number of texts sent to each multi-process worker at a time.
            dtype: Model dtype. Defaults to float16 on CUDA workers.
            max_seq_length: Optional max sequence length for embedding inputs.
        """
        self.model_name = self._resolve_model_path(model_name)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.pool = None
        self.model = None

        if devices is not None:
            self.device = list(devices)
        elif device is None:
            self.device = (
                "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = device
        else:
            self.device = list(device)

        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers not installed. MultilingualEmbedder will not work."
            )
            return

        logger.info(
            "Initialized embedding coordinator for %s; encode target device(s): %s",
            self.model_name,
            self._get_encode_devices(),
        )

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

    def _get_encode_devices(self) -> list[str]:
        """Return target devices for encoding, using all CUDA GPUs when available."""
        if isinstance(self.device, list):
            return self.device

        if (
            (self.device == "cuda" or self.device.startswith("cuda:"))
            and torch is not None
            and torch.cuda.is_available()
        ):
            device_count = torch.cuda.device_count()
            if device_count > 1:
                return [f"cuda:{idx}" for idx in range(device_count)]
            return ["cuda:0"]

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

    def _ensure_single_process_model(self, device: str) -> None:
        if self.model is not None:
            return
        try:
            self.model = _load_sentence_transformer(
                self.model_name,
                device=device,
                dtype=self.dtype,
                max_seq_length=self.max_seq_length,
            )
            logger.info("Loaded embedding model %s on %s", self.model_name, device)
        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", self.model_name, e)
            self.model = None
            raise

    def start_multi_process_pool(
        self, target_devices: list[str] | None = None
    ) -> EmbeddingWorkerPool | None:
        """Start a persistent multi-process pool for encoding.

        Args:
            target_devices: List of devices (e.g., ['cuda:0', 'cuda:1']).
                Defaults to all available GPUs.

        Returns:
            The created worker pool, or None when only one device is selected.
        """
        if self.pool is not None:
            return self.pool

        if SentenceTransformer is None:
            raise RuntimeError("Embedder model not loaded.")

        if target_devices is None:
            target_devices = self._get_encode_devices()

        if len(target_devices) <= 1:
            logger.info(
                "Only one device detected (%s). Skipping pool creation.", target_devices
            )
            return None

        logger.info("Starting persistent multi-process pool on %s", target_devices)
        self.pool = EmbeddingWorkerPool(
            model_name=self.model_name,
            devices=target_devices,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            max_seq_length=self.max_seq_length,
        )
        self.pool.start()
        return self.pool

    def stop_multi_process_pool(self) -> None:
        """Stop the active multi-process pool and clear CUDA cache."""
        if self.pool is not None:
            logger.info("Stopping multi-process pool...")
            self.pool.stop()
            self.pool = None
            self._clear_cuda_cache()

    def _single_process_encode(
        self,
        texts: list[str],
        prefix: str,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode on one device, preferring SentenceTransformers prompt support."""
        devices = self._get_encode_devices()
        self._ensure_single_process_model(devices[0])
        return _encode_with_prompt(
            self.model,
            texts,
            prefix,
            batch_size=self.batch_size,
            show_progress=show_progress,
        )

    def _multi_process_encode(
        self,
        texts: list[str],
        prefix: str,
        devices: list[str],
        show_progress: bool,
    ) -> np.ndarray:
        """Encode using the custom spawned worker pool."""
        pool_to_use = self.pool
        own_pool = False

        if pool_to_use is None:
            logger.info(
                "Starting temporary multi-process pool across %s device(s): %s",
                len(devices),
                devices,
            )
            pool_to_use = EmbeddingWorkerPool(
                model_name=self.model_name,
                devices=devices,
                batch_size=self.batch_size,
                chunk_size=self.chunk_size,
                dtype=self.dtype,
                max_seq_length=self.max_seq_length,
            )
            pool_to_use.start()
            own_pool = True

        try:
            return pool_to_use.encode(
                texts,
                prefix=prefix,
                show_progress=show_progress,
                chunk_size=self.chunk_size,
            )
        finally:
            if own_pool:
                pool_to_use.stop()
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
        if SentenceTransformer is None:
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
