"""BM25 indexing and search for legal document corpora."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
from bm25s.tokenization import Tokenizer

CORPUS_FILENAME = "corpus.jsonl"
OFFSETS_FILENAME = "doc_offsets.npy"
METADATA_FILENAME = "omnilex_metadata.json"


class BM25Index:
    """BM25 index for keyword search over legal documents.

    The implementation uses ``bm25s`` sparse arrays for retrieval and stores
    document payloads separately as JSONL so large indices can be loaded with
    memory-mapped scoring arrays in offline Kaggle inference.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        *,
        mmap: bool = True,
    ):
        """Initialize BM25 index.

        Args:
            documents: Optional in-memory document dictionaries to index.
            text_field: Key for document text in dict.
            citation_field: Key for citation string in dict.
            mmap: Whether loaded on-disk indices should memory-map arrays.
        """
        self.text_field = text_field
        self.citation_field = citation_field
        self.mmap = mmap

        self.documents: list[dict[str, Any]] = []
        self.index: bm25s.BM25 | None = None
        self.tokenizer = self._create_tokenizer()
        self.index_path: Path | None = None
        self.corpus_path: Path | None = None
        self._doc_offsets: np.ndarray | None = None

        if documents:
            self.build(documents)

    @staticmethod
    def _create_tokenizer() -> Tokenizer:
        """Create a tokenizer matching the previous lowercase word-token behavior."""
        return Tokenizer(
            stemmer=None,
            stopwords=None,
            splitter=r"\w+",
            lower=True,
        )

    @staticmethod
    def _create_retriever() -> bm25s.BM25:
        return bm25s.BM25(
            method="lucene",
            dtype="float32",
            int_dtype="int32",
            backend="numpy",
            csc_backend="numpy",
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using the same simple lowercase word splitter as the index."""
        return re.findall(r"\w+", text.lower())

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from in-memory documents."""
        self.documents = documents
        self.index_path = None
        self.corpus_path = None
        self._doc_offsets = None
        self.tokenizer = self._create_tokenizer()

        texts = [self._get_text(doc) for doc in documents]
        tokenized = self.tokenizer.tokenize(
            texts,
            update_vocab=True,
            return_as="tuple",
            allow_empty=True,
            show_progress=True,
        )

        self.index = self._create_retriever()
        self.index.index(tokenized, show_progress=True)

    @classmethod
    def build_from_iterable(
        cls,
        documents: Iterable[dict[str, Any]],
        output_dir: Path | str,
        *,
        text_field: str = "text",
        citation_field: str = "citation",
        length: int | None = None,
    ) -> BM25Index:
        """Build and save a BM25 index from a document iterable.

        This path keeps the corpus payload on disk and avoids storing a second
        tokenized corpus on the ``BM25Index`` wrapper.
        """
        instance = cls(text_field=text_field, citation_field=citation_field)
        instance.build_to_disk(
            documents,
            output_dir,
            length=length,
        )
        return instance

    def build_to_disk(
        self,
        documents: Iterable[dict[str, Any]],
        output_dir: Path | str,
        *,
        length: int | None = None,
    ) -> None:
        """Build the index from an iterable and immediately persist it to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = output_dir / CORPUS_FILENAME
        offsets_path = output_dir / OFFSETS_FILENAME
        metadata_path = output_dir / METADATA_FILENAME

        offsets: list[int] = []

        with open(corpus_path, "wb") as f:
            for raw_doc in documents:
                doc = self._normalize_doc(raw_doc)
                offsets.append(f.tell())
                line = json.dumps(doc, ensure_ascii=False) + "\n"
                f.write(line.encode("utf-8"))

        if not offsets:
            raise ValueError("Cannot build a BM25 index from an empty corpus.")

        offsets_array = np.asarray(offsets, dtype=np.int64)
        num_docs = int(len(offsets_array))
        np.save(offsets_path, offsets_array)

        self.documents = []
        self.index_path = output_dir
        self.corpus_path = corpus_path
        self._doc_offsets = offsets_array
        self.tokenizer = self._create_tokenizer()

        tokenized = self.tokenizer.tokenize(
            self._iter_corpus_texts(corpus_path),
            update_vocab=True,
            return_as="tuple",
            allow_empty=True,
            length=length or num_docs,
            show_progress=True,
        )

        self.index = self._create_retriever()
        self.index.index(tokenized, show_progress=True)
        self.index.save(output_dir)
        self.tokenizer.save_vocab(output_dir)
        self.tokenizer.save_stopwords(output_dir)

        metadata = {
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "num_docs": num_docs,
            "corpus_filename": CORPUS_FILENAME,
            "offsets_filename": OFFSETS_FILENAME,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str, *, mmap: bool = True) -> BM25Index:
        """Load an index from a native ``bm25s`` index directory."""
        path = Path(path)
        metadata_path = path / METADATA_FILENAME

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"BM25 metadata not found at {metadata_path}. "
                "Rebuild the index with the bm25s-based builder."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        instance = cls(
            text_field=metadata.get("text_field", "text"),
            citation_field=metadata.get("citation_field", "citation"),
            mmap=mmap,
        )
        instance.index_path = path
        instance.corpus_path = path / metadata.get("corpus_filename", CORPUS_FILENAME)
        offsets_path = path / metadata.get("offsets_filename", OFFSETS_FILENAME)

        instance.index = bm25s.BM25.load(path, mmap=mmap, load_corpus=False)
        instance.tokenizer = instance._create_tokenizer()
        instance.tokenizer.load_vocab(path)
        instance.tokenizer.load_stopwords(path)
        instance._doc_offsets = np.load(
            offsets_path,
            mmap_mode="r" if mmap else None,
        )

        return instance

    def save(self, path: Path | str) -> None:
        """Save the index to a native ``bm25s`` directory."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        path = Path(path)
        if self.documents:
            self.build_to_disk(self.documents, path, length=len(self.documents))
            return

        if self.index_path == path:
            return

        raise ValueError(
            "This BM25Index is already disk-backed. Rebuild it at the target path "
            "instead of copying sparse index files through save()."
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the index with a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        if top_k <= 0:
            return []

        query_tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple",
            allow_empty=False,
            show_progress=False,
        )

        if not query_tokens.ids or not query_tokens.ids[0]:
            return []

        num_docs = int(self.index.scores["num_docs"])
        k = min(top_k, num_docs)
        if k <= 0:
            return []

        retrieved = self.index.retrieve(
            query_tokens,
            k=k,
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )

        results: list[dict[str, Any]] = []
        for doc_id, score in zip(
            retrieved.documents[0],
            retrieved.scores[0],
            strict=False,
        ):
            score_value = float(score)
            if score_value <= 0:
                continue

            doc = self._get_doc(int(doc_id))
            if return_scores:
                doc["_score"] = score_value
            results.append(doc)

        return results

    @property
    def num_docs(self) -> int:
        if self.index is not None:
            return int(self.index.scores["num_docs"])
        return len(self.documents)

    def _get_doc(self, doc_id: int) -> dict[str, Any]:
        if self.documents:
            return self.documents[doc_id].copy()

        if self.corpus_path is None or self._doc_offsets is None:
            raise ValueError("Document corpus is not available for this BM25 index.")

        with open(self.corpus_path, "rb") as f:
            f.seek(int(self._doc_offsets[doc_id]))
            return json.loads(f.readline().decode("utf-8"))

    def _get_text(self, doc: dict[str, Any]) -> str:
        value = doc.get(self.text_field, "")
        if value is None:
            return ""
        return str(value)

    def _normalize_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(doc)
        normalized[self.text_field] = self._get_text(normalized)
        citation = normalized.get(self.citation_field, "")
        normalized[self.citation_field] = "" if citation is None else str(citation)
        return normalized

    def _iter_corpus_texts(self, corpus_path: Path) -> Iterator[str]:
        with open(corpus_path, "rb") as f:
            for line in f:
                yield self._get_text(json.loads(line.decode("utf-8")))


def build_index(
    documents: list[dict[str, Any]],
    text_field: str = "text",
    citation_field: str = "citation",
) -> BM25Index:
    """Build a BM25 index from documents."""
    return BM25Index(
        documents=documents,
        text_field=text_field,
        citation_field=citation_field,
    )


def search(
    index: BM25Index,
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search an index with a query."""
    return index.search(query, top_k=top_k)


def iter_jsonl_corpus(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream a corpus from a JSONL file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_corpus(path: Path | str) -> list[dict[str, Any]]:
    """Load a corpus from a JSONL file."""
    return list(iter_jsonl_corpus(path))


def save_jsonl_corpus(documents: list[dict[str, Any]], path: Path | str) -> None:
    """Save a corpus to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
