#!/usr/bin/env python3
"""Build BM25 indices for legal document corpora.

Creates searchable indices for:
1. Federal laws (SR corpus)
2. Court decisions (BGE corpus)

Usage:
    python scripts/build_indices.py
    python scripts/build_indices.py --input-dir ./data/raw --output-dir ./data/processed
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index, iter_jsonl_corpus


def build_laws_index(input_dir: Path, output_dir: Path) -> None:
    """Build BM25 index for federal laws corpus.

    Args:
        input_dir: Directory containing raw corpus files
        output_dir: Directory to save index
    """
    print("Building federal laws index...")

    # Try to find laws corpus file
    possible_paths = [
        input_dir / "samples" / "federal_laws.jsonl",
        input_dir / "federal_laws.jsonl",
        input_dir / "laws" / "federal_laws.jsonl",
        input_dir / "corpus" / "laws.jsonl",
    ]

    corpus_path = None
    for path in possible_paths:
        if path.exists():
            corpus_path = path
            break

    if corpus_path is None:
        print("  Warning: No federal laws corpus found. Skipping index build.")
        print(f"  Expected one of: {[str(p) for p in possible_paths]}")
        return

    # Build index
    output_path = output_dir / "laws_index"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Streaming corpus from {corpus_path}")
    try:
        index = BM25Index.build_from_iterable(
            iter_jsonl_corpus(corpus_path),
            output_path,
            text_field="text",
            citation_field="citation",
        )
    except ValueError as exc:
        print(f"  Warning: {exc}. Skipping index build.")
        return

    # Test search
    test_query = "Vertrag"
    results = index.search(test_query, top_k=3)
    print(f"  Index saved to {output_path}")
    print(f"  Test search for '{test_query}': {len(results)} results")


def build_courts_index(input_dir: Path, output_dir: Path) -> None:
    """Build BM25 index for court decisions corpus.

    Args:
        input_dir: Directory containing raw corpus files
        output_dir: Directory to save index
    """
    print("\nBuilding court decisions index...")

    # Try to find courts corpus file
    possible_paths = [
        input_dir / "samples" / "court_decisions.jsonl",
        input_dir / "court_decisions.jsonl",
        input_dir / "courts" / "bge.jsonl",
        input_dir / "corpus" / "courts.jsonl",
    ]

    corpus_path = None
    for path in possible_paths:
        if path.exists():
            corpus_path = path
            break

    if corpus_path is None:
        print("  Warning: No court decisions corpus found. Skipping index build.")
        print(f"  Expected one of: {[str(p) for p in possible_paths]}")
        return

    # Build index
    output_path = output_dir / "courts_index"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Streaming corpus from {corpus_path}")
    try:
        index = BM25Index.build_from_iterable(
            iter_jsonl_corpus(corpus_path),
            output_path,
            text_field="text",
            citation_field="citation",
        )
    except ValueError as exc:
        print(f"  Warning: {exc}. Skipping index build.")
        return

    # Test search
    test_query = "Meinungsfreiheit"
    results = index.search(test_query, top_k=3)
    print(f"  Index saved to {output_path}")
    print(f"  Test search for '{test_query}': {len(results)} results")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 indices for legal corpora")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with corpus files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for indices",
    )
    parser.add_argument(
        "--laws-only",
        action="store_true",
        help="Only build laws index",
    )
    parser.add_argument(
        "--courts-only",
        action="store_true",
        help="Only build courts index",
    )

    args = parser.parse_args()

    # Build indices
    if not args.courts_only:
        build_laws_index(args.input_dir, args.output_dir)

    if not args.laws_only:
        build_courts_index(args.input_dir, args.output_dir)

    print("\nIndex building complete!")


if __name__ == "__main__":
    main()
