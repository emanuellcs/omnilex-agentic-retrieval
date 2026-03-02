"""Pytest fixtures for omnilex tests."""

import json
import tempfile
from pathlib import Path

import pytest

from omnilex.citations.sample_data import SAMPLE_COURTS, SAMPLE_LAWS


@pytest.fixture
def sample_sr_citations():
    """Sample federal law citations for testing.

    Note: Citations are normalized to paragraph level only (no lit., Ziff., etc.)
    """
    return [
        "Art. 1 ZGB",
        "Art. 11 OR",
        "Art. 117 StGB",
        "Art. 1 Abs. 2 ZGB",
        "Art. 41 Abs. 1 OR",
    ]


@pytest.fixture
def sample_bge_citations():
    """Sample BGE (court decision) citations for testing."""
    return [
        "BGE 116 Ia 56 E. 2b",
        "BGE 119 II 449 E. 3.4",
        "BGE 121 III 38 E. 2b",
        "BGE 145 II 32 E. 3.1",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return {
        "query_id": "test_0001",
        "query": "What are the requirements for a valid contract under Swiss law?",
        "gold_citations": ["Art. 1 OR", "Art. 11 OR", "BGE 119 II 449 E. 3.4"],
    }


@pytest.fixture
def sample_predictions():
    """Sample predictions for evaluation testing."""
    return [
        ["Art. 1 ZGB", "BGE 116 Ia 56 E. 2b"],
        ["Art. 1 OR", "Art. 11 OR"],
        ["BGE 121 III 38 E. 2b"],
    ]


@pytest.fixture
def sample_gold():
    """Sample gold labels for evaluation testing."""
    return [
        ["Art. 1 ZGB", "BGE 116 Ia 56 E. 2b"],  # Perfect match
        ["Art. 1 OR", "BGE 119 II 449 E. 3.4"],  # Partial match
        ["Art. 117 StGB", "BGE 121 III 38 E. 2b"],  # Partial match
    ]


@pytest.fixture
def sample_laws_corpus():
    """Sample federal laws corpus for testing (from shared sample_data module)."""
    return SAMPLE_LAWS


@pytest.fixture
def sample_courts_corpus():
    """Sample court decisions corpus for testing (from shared sample_data module)."""
    return SAMPLE_COURTS


@pytest.fixture
def temp_corpus_file(sample_laws_corpus):
    """Create a temporary corpus file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc in sample_laws_corpus:
            f.write(json.dumps(doc) + "\n")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_submission_file():
    """Create a temporary submission file for testing."""
    import csv

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "test_0001",
                "predicted_citations": "Art. 1 ZGB;BGE 116 Ia 56 E. 2b",
            }
        )
        writer.writerow(
            {
                "query_id": "test_0002",
                "predicted_citations": "Art. 1 OR",
            }
        )
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_gold_file():
    """Create a temporary gold file for testing."""
    import csv

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "gold_citations"])
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "test_0001",
                "gold_citations": "Art. 1 ZGB;BGE 116 Ia 56 E. 2b",
            }
        )
        writer.writerow(
            {
                "query_id": "test_0002",
                "gold_citations": "Art. 1 OR;BGE 119 II 449 E. 3.4",
            }
        )
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()
