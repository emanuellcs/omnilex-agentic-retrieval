#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_csv_documents(csv_path: str, max_rows: int | None, chunksize: int):
    rows_seen = 0
    reader = pd.read_csv(
        csv_path,
        usecols=["citation", "text"],
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk = chunk.fillna("")
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        rows_seen += len(chunk)
        for citation, text in chunk[["citation", "text"]].itertuples(
            index=False,
            name=None,
        ):
            yield {"citation": str(citation), "text": str(text)}


def build_bm25_indices(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Laws
    if args.laws_csv:
        logger.info(f"Building BM25 index for laws from {args.laws_csv}")

        docs = iter_csv_documents(
            args.laws_csv,
            max_rows=args.max_rows_laws,
            chunksize=args.chunksize,
        )

        save_path = output_dir / "laws_index"
        BM25Index.build_from_iterable(docs, save_path)
        logger.info(f"Laws BM25 index saved to {save_path}")

    # 2. Process Courts
    if args.courts_csv:
        logger.info(f"Building BM25 index for courts from {args.courts_csv}")

        docs = iter_csv_documents(
            args.courts_csv,
            max_rows=args.max_rows_courts,
            chunksize=args.chunksize,
        )

        save_path = output_dir / "courts_index"
        BM25Index.build_from_iterable(docs, save_path)
        logger.info(f"Courts BM25 index saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build BM25 indices for legal corpora (CSV version)"
    )
    parser.add_argument("--laws-csv", type=str, help="Path to laws_de.csv")
    parser.add_argument(
        "--courts-csv", type=str, help="Path to court_considerations.csv"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument("--max-rows-laws", type=int, help="Limit laws rows for testing")
    parser.add_argument(
        "--max-rows-courts", type=int, help="Limit courts rows for testing"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="CSV rows to read per chunk while streaming input",
    )

    args = parser.parse_args()
    build_bm25_indices(args)


if __name__ == "__main__":
    main()
