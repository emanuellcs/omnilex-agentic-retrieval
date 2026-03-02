#!/usr/bin/env python3
"""Download and prepare LEXam data from HuggingFace.

Downloads:
1. LEXam dataset (training data with queries and gold citations)
2. Swiss legal corpora (laws and court decisions)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output-dir ./data/raw
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Add src to path for importing omnilex
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from omnilex.citations.abbreviations import get_german_abbreviations
from omnilex.citations.normalizer import CitationNormalizer
from omnilex.citations.sample_data import (
    SAMPLE_COURTS,
    SAMPLE_LAWS,
    SAMPLE_SUBMISSION,
    SAMPLE_TEST_QUERIES,
    SAMPLE_TRAIN_QUERIES,
)


def load_valid_court_citations(output_dir: Path) -> set[str]:
    """Load valid court citation IDs from court_considerations.csv.

    Returns a set of canonical court citation IDs that exist in the retrieval corpus.
    Only includes citations with consideration references (E. X).
    Supports both BGE citations (e.g., BGE 139 I 2 E. 5.1) and docket-style
    citations (e.g., 5A_800/2019 E. 2, 2C_123/2020 E. 1.2.3).
    """
    valid_citations = set()
    court_file = output_dir / "retrieval" / "court_considerations.csv"

    if not court_file.exists():
        print(f"  Warning: {court_file} not found. Skipping court citation filtering.")
        return valid_citations

    print(f"  Loading valid court citations from {court_file}...")
    with open(court_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citation = row["citation"]
            # Only include citations with consideration references (E. X)
            if " E. " in citation:
                valid_citations.add(citation)

    print(f"  Loaded {len(valid_citations)} valid court citation patterns (with E.)")
    return valid_citations


def filter_court_citations(citations: list[str], valid_court: set[str]) -> list[str]:
    """Filter court citations to only those with considerations (E.) in corpus.

    Args:
        citations: List of canonical citation IDs
        valid_court: Set of valid court citations from corpus (only those with E.)

    Returns:
        Filtered list - keeps law citations, removes court citations without E. or not in corpus
    """
    filtered = []
    for c in citations:
        # Check if it's a court citation (BGE or docket-style like 5A_800/2019)
        is_court_citation = c.startswith("BGE") or re.match(r"^\d+[A-Z]_", c)
        if is_court_citation:
            # Only keep court citations that have E. AND are in the valid set
            if " E. " in c and c in valid_court:
                filtered.append(c)
        else:
            # Keep all law citations
            filtered.append(c)
    return filtered


def extract_citations_from_text(text: str, normalizer: CitationNormalizer) -> list[str]:
    """Extract and normalize legal citations from text.

    Args:
        text: Text containing potential legal citations
        normalizer: CitationNormalizer instance

    Returns:
        List of unique canonical citation IDs
    """
    if not text:
        return []

    citations = []

    # Build law abbreviations regex from JSON file (single source of truth)
    # Sorted by length (longest first) to match longer abbreviations first
    all_abbrevs = get_german_abbreviations()
    law_abbrevs = r"(?:" + "|".join(all_abbrevs) + r")"

    # Pattern for Article citations (Art. X [Abs. Y] LAW)
    art_pattern = (
        rf"Art\.?\s*(\d+[a-z]?)"
        rf"(?:\s+(?:Abs\.?|al\.?|cpv\.?)\s*(\d+[a-z]?))?"
        rf"(?:\s+(?:Ziff\.?|lit\.?|Bst\.?)\s*(\d*[a-z]?))?"
        rf"\s+{law_abbrevs}"
    )

    # Find Article citations
    for match in re.finditer(art_pattern, text, re.IGNORECASE):
        # Extract the full match including the law abbreviation
        start = match.start()
        end = match.end()
        citation_text = text[start:end]
        normalized = normalizer.normalize(citation_text)
        if normalized:
            citations.append(normalized.canonical_id)

    # Use the BGE pattern from the normalizer (single source of truth)
    for match in re.finditer(CitationNormalizer.BGE_PATTERN, text, re.IGNORECASE):
        citation_text = match.group(0)
        normalized = normalizer.normalize(citation_text)
        if normalized:
            citations.append(normalized.canonical_id)

    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return unique_citations


def download_lexam(output_dir: Path, filter_by_corpus: bool = False) -> None:
    """Download LEXam dataset from HuggingFace and extract citations.

    LEXam is a legal exam dataset. We use the 'open_question' config which
    contains questions with full-text answers. Citations are extracted from
    both questions and answers using the CitationNormalizer.

    Only entries with at least one citation are kept. The data is then
    split 70/30 into train/val sets.

    Args:
        output_dir: Directory to save the processed data
        filter_by_corpus: If True, filter court citations to only those in the corpus
    """
    import random

    print("Downloading LEXam dataset (open_question config)...")

    try:
        # Load the open_question config (has both questions and answers)
        dataset = load_dataset("LEXam-Benchmark/LEXam", "open_question")

        normalizer = CitationNormalizer()
        lexam_dir = output_dir / "lexam"
        lexam_dir.mkdir(parents=True, exist_ok=True)

        # Collect all items with citations across all HuggingFace splits
        all_with_citations = []
        total_processed = 0

        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"  Processing {split_name} split ({len(split_data)} samples)...")

            for i, item in enumerate(tqdm(split_data, desc=f"  {split_name}")):
                question = item.get("question", "")
                answer = item.get("answer", "")

                # Extract citations from both question and answer
                full_text = f"{question} {answer}"
                citations = extract_citations_from_text(full_text, normalizer)

                total_processed += 1

                # Only keep entries with citations
                if citations:
                    processed = {
                        "query_id": "",  # Will be assigned after split
                        "query": question,
                        "gold_citations": ";".join(citations),
                    }
                    all_with_citations.append(processed)

        print(f"\n  Total processed: {total_processed}")
        print(f"  Entries with citations: {len(all_with_citations)}")

        # Filter court citations if requested
        if filter_by_corpus:
            valid_court = load_valid_court_citations(output_dir)
            if valid_court:
                filtered_entries = []
                total_court_removed = 0
                entries_removed = 0

                for item in all_with_citations:
                    citations = item["gold_citations"].split(";")
                    original_count = len(citations)
                    filtered = filter_court_citations(citations, valid_court)
                    total_court_removed += original_count - len(filtered)

                    if filtered:
                        item["gold_citations"] = ";".join(filtered)
                        filtered_entries.append(item)
                    else:
                        entries_removed += 1

                all_with_citations = filtered_entries
                print("\n  Filtering by corpus:")
                print(f"    - Court citations removed: {total_court_removed}")
                print(f"    - Entries removed (no citations left): {entries_removed}")
                print(f"    - Entries remaining: {len(all_with_citations)}")

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_with_citations)

        # Assign enumerated IDs (train_0001, train_0002, etc.)
        for i, item in enumerate(all_with_citations, start=1):
            item["query_id"] = f"train_{i:04d}"

        # Save as single train.csv
        csv_path = lexam_dir / "train.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["query_id", "query", "gold_citations"]
            )
            writer.writeheader()
            for item in all_with_citations:
                writer.writerow(
                    {
                        "query_id": item["query_id"],
                        "query": item["query"],
                        "gold_citations": item["gold_citations"],
                    }
                )

        print(f"\nLEXam saved to {lexam_dir}")
        print(f"  - train.csv: {len(all_with_citations)} entries with citations")

    except Exception as e:
        print(f"Warning: Could not download LEXam: {e}")
        print("You may need to request access to the dataset on HuggingFace.")


def download_swiss_citations(output_dir: Path) -> None:
    """Download Swiss citation extraction dataset.

    Contains Swiss legal texts with citation annotations.
    """
    print("\nDownloading Swiss citation dataset...")

    try:
        dataset = load_dataset("rcds/swiss_citation_extraction")

        for split_name in dataset.keys():
            split_data = dataset[split_name]
            output_path = output_dir / "swiss_citations" / f"{split_name}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Saving {split_name} split ({len(split_data)} samples)...")

            with open(output_path, "w", encoding="utf-8") as f:
                for item in tqdm(split_data, desc=f"  {split_name}"):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Swiss citations saved to {output_dir / 'swiss_citations'}")

    except Exception as e:
        print(f"Warning: Could not download Swiss citations: {e}")


def create_sample_data(output_dir: Path) -> None:
    """Create sample data files for testing.

    Creates minimal sample files that can be used for development
    without downloading full datasets.

    Uses shared sample data from omnilex.citations.sample_data module.
    """
    print("\nCreating sample data files...")

    # Write sample files
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Write train.csv
    with open(samples_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "gold_citations"])
        writer.writeheader()
        for item in SAMPLE_TRAIN_QUERIES:
            writer.writerow(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "gold_citations": item["gold_citations"],
                }
            )

    # Write test.csv
    with open(samples_dir / "test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query"])
        writer.writeheader()
        writer.writerows(SAMPLE_TEST_QUERIES)

    # Write sample_submission.csv (also to data/ root)
    with open(
        samples_dir / "sample_submission.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        writer.writerows(SAMPLE_SUBMISSION)

    # Also write to data/ root
    data_root = output_dir.parent
    with open(
        data_root / "sample_submission.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        writer.writerows(SAMPLE_SUBMISSION)

    # Write corpus JSONL files
    with open(samples_dir / "federal_laws.jsonl", "w", encoding="utf-8") as f:
        for doc in SAMPLE_LAWS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(samples_dir / "court_decisions.jsonl", "w", encoding="utf-8") as f:
        for doc in SAMPLE_COURTS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Sample data saved to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download competition data from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--skip-lexam",
        action="store_true",
        help="Skip downloading LEXam dataset",
    )
    parser.add_argument(
        "--skip-citations",
        action="store_true",
        help="Skip downloading Swiss citations dataset",
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Only create sample data (no downloads)",
    )
    parser.add_argument(
        "--filter-by-corpus",
        action="store_true",
        help="Filter court citations to only those present in the corpus",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples_only:
        create_sample_data(args.output_dir)
        return

    # Download datasets
    if not args.skip_lexam:
        download_lexam(args.output_dir, filter_by_corpus=args.filter_by_corpus)

    if not args.skip_citations:
        download_swiss_citations(args.output_dir)

    # Always create sample data for development
    create_sample_data(args.output_dir)

    print("\nData download complete!")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
