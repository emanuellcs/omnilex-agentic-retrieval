#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.dense_retrieval import MultilingualEmbedder, FAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indices(args):
    embedder = MultilingualEmbedder(
        model_name=args.model_name, batch_size=args.batch_size
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Laws
    if args.laws_csv:
        logger.info(f"Processing laws from {args.laws_csv}")
        df_laws = pd.read_csv(args.laws_csv)
        if args.max_rows_laws:
            df_laws = df_laws.head(args.max_rows_laws)

        texts = df_laws["text"].tolist()
        logger.info(f"Encoding {len(texts)} law passages...")
        embeddings = embedder.encode(texts, is_query=False)

        index = FAISSIndex()
        docs = df_laws[["citation", "text"]].to_dict("records")
        index.build(embeddings, docs)

        save_path = output_dir / "laws_faiss"
        index.save(save_path)
        logger.info(f"Laws FAISS index saved to {save_path}.faiss and .pkl")

    # 2. Process Courts (Chunked)
    if args.courts_csv:
        logger.info(f"Processing courts from {args.courts_csv}")

        all_embeddings = []
        all_docs = []

        # Load in chunks to avoid OOM
        chunksize = 50000
        reader = pd.read_csv(args.courts_csv, chunksize=chunksize)

        row_count = 0
        for chunk in reader:
            if args.max_rows_courts and row_count >= args.max_rows_courts:
                break

            if args.max_rows_courts:
                current_chunk = chunk.head(args.max_rows_courts - row_count)
            else:
                current_chunk = chunk

            texts = current_chunk["text"].fillna("").tolist()
            logger.info(f"Encoding chunk of {len(texts)} court passages...")
            chunk_embeddings = embedder.encode(texts, is_query=False)

            all_embeddings.append(chunk_embeddings)
            all_docs.extend(current_chunk[["citation", "text"]].to_dict("records"))

            row_count += len(current_chunk)
            if args.max_rows_courts and row_count >= args.max_rows_courts:
                break

        final_embeddings = np.vstack(all_embeddings)

        index = FAISSIndex()
        # Use IVFFlat for large court corpus if not limited
        index_type = "IVFFlat" if len(all_docs) > 100000 else "Flat"
        index.build(final_embeddings, all_docs, index_type=index_type)

        save_path = output_dir / "courts_faiss"
        index.save(save_path)
        logger.info(f"Courts FAISS index saved to {save_path}.faiss and .pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for legal corpora"
    )
    parser.add_argument("--laws-csv", type=str, help="Path to laws_de.csv")
    parser.add_argument(
        "--courts-csv", type=str, help="Path to court_considerations.csv"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--model-name", type=str, default="intfloat/multilingual-e5-large"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows-laws", type=int, help="Limit laws rows for testing")
    parser.add_argument(
        "--max-rows-courts", type=int, help="Limit courts rows for testing"
    )

    args = parser.parse_args()
    build_indices(args)


if __name__ == "__main__":
    main()
