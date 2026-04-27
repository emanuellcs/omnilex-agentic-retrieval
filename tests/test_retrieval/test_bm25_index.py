import json

import numpy as np

from omnilex.retrieval.bm25_index import (
    CORPUS_FILENAME,
    METADATA_FILENAME,
    OFFSETS_FILENAME,
    BM25Index,
)


def sample_docs():
    return [
        {"citation": "A", "text": "Vertrag Kauf Vertrag"},
        {"citation": "B", "text": "Meinungsfreiheit und Gericht"},
        {"citation": "C", "text": "Scheidung Ehe Familie"},
    ]


def test_search_returns_ranked_documents_with_scores():
    index = BM25Index(sample_docs())

    results = index.search("vertrag", top_k=2, return_scores=True)

    assert [doc["citation"] for doc in results] == ["A"]
    assert results[0]["_score"] > 0


def test_empty_query_and_top_k_clamping():
    index = BM25Index(sample_docs())

    assert index.search("", top_k=3) == []
    assert index.search("vertrag", top_k=99)[0]["citation"] == "A"
    assert index.search("vertrag", top_k=0) == []


def test_save_load_uses_disk_artifacts_and_mmap(tmp_path):
    index_path = tmp_path / "laws_index"
    index = BM25Index(sample_docs())
    index.save(index_path)

    expected_files = {
        CORPUS_FILENAME,
        OFFSETS_FILENAME,
        METADATA_FILENAME,
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "vocab.index.json",
    }
    assert expected_files.issubset({path.name for path in index_path.iterdir()})

    loaded = BM25Index.load(index_path, mmap=True)
    results = loaded.search("gericht", top_k=2, return_scores=True)

    assert isinstance(loaded.index.scores["data"], np.memmap)
    assert results[0]["citation"] == "B"
    assert results[0]["_score"] > 0


def test_build_from_iterable_streams_corpus_payload(tmp_path):
    index_path = tmp_path / "courts_index"
    BM25Index.build_from_iterable(iter(sample_docs()), index_path)

    metadata = json.loads((index_path / METADATA_FILENAME).read_text())
    offsets = np.load(index_path / OFFSETS_FILENAME)
    loaded = BM25Index.load(index_path)

    assert metadata["num_docs"] == 3
    assert offsets.shape == (3,)
    assert loaded.num_docs == 3
    assert loaded.search("familie", top_k=5)[0]["citation"] == "C"
