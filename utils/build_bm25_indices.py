#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_bm25_indices(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Laws
    if args.laws_csv:
        logger.info(f"Building BM25 index for laws from {args.laws_csv}")
        df_laws = pd.read_csv(args.laws_csv)
        if args.max_rows_laws:
            df_laws = df_laws.head(args.max_rows_laws)

        docs = df_laws[["citation", "text"]].to_dict("records")
        index = BM25Index(documents=docs)

        save_path = output_dir / "laws_index.pkl"
        index.save(save_path)
        logger.info(f"Laws BM25 index saved to {save_path}")

    # 2. Process Courts
    if args.courts_csv:
        logger.info(f"Building BM25 index for courts from {args.courts_csv}")

        # NOTE: BM25 requires all documents in memory.
        # For 2.5M rows, this might be tight.
        # We'll load in a single read_csv if possible, or limit it for local dev.

        if args.max_rows_courts:
            df_courts = pd.read_csv(args.courts_csv, nrows=args.max_rows_courts)
        else:
            # Try to load everything, but be careful
            # We use usecols to save RAM
            df_courts = pd.read_csv(args.courts_csv, usecols=["citation", "text"])

        docs = df_courts[["citation", "text"]].to_dict("records")
        index = BM25Index(documents=docs)

        save_path = output_dir / "courts_index.pkl"
        index.save(save_path)
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

    args = parser.parse_args()
    build_bm25_indices(args)


if __name__ == "__main__":
    main()
