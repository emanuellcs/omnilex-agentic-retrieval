#!/usr/bin/env python3
import argparse
import logging
import sys
import gc
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

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

    # Start persistent pool for multiple GPUs
    embedder.start_multi_process_pool()

    try:
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

            logger.info("Building FAISS index for laws...")
            index.build(embeddings, docs)

            save_path = output_dir / "laws_faiss"
            index.save(save_path)
            logger.info(f"Laws FAISS index saved to {save_path}.faiss and .pkl")

            # Clean up aggressively
            del embeddings, docs, df_laws, texts
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Process Courts (Chunked & Incremental)
        if args.courts_csv:
            logger.info(f"Processing courts from {args.courts_csv}")

            index = FAISSIndex()
            is_trained = False

            # Load in chunks to avoid RAM OOM
            chunksize = 50000
            reader = pd.read_csv(args.courts_csv, chunksize=chunksize)

            row_count = 0
            for chunk in tqdm(reader, desc="Processing court chunks"):
                if args.max_rows_courts and row_count >= args.max_rows_courts:
                    break

                if args.max_rows_courts:
                    current_chunk = chunk.head(args.max_rows_courts - row_count)
                else:
                    current_chunk = chunk

                texts = current_chunk["text"].fillna("").tolist()
                logger.info(
                    f"Encoding chunk of {len(texts)} court passages (Total processed: {row_count})..."
                )
                chunk_embeddings = embedder.encode(texts, is_query=False)

                current_docs = current_chunk[["citation", "text"]].to_dict("records")

                if not is_trained:
                    # IVFFlat is recommended for >100k docs.
                    # Training on 50k is sufficient for a 2.5M corpus.
                    logger.info("Training IVFFlat index on first chunk...")
                    index.train(
                        chunk_embeddings,
                        index_type="IVFFlat",
                        total_expected_docs=2500000,
                    )
                    is_trained = True

                index.add_batch(chunk_embeddings, current_docs)
                row_count += len(current_chunk)

                # Aggressive memory hygiene: delete everything before next iteration
                del chunk_embeddings, current_docs, current_chunk, texts
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Report GPU usage if possible
                if torch and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem = torch.cuda.memory_allocated(i) / 1024**3
                        logger.info(f"GPU {i} memory allocated: {mem:.2f} GB")

                if args.max_rows_courts and row_count >= args.max_rows_courts:
                    break

            save_path = output_dir / "courts_faiss"
            index.save(save_path)
            logger.info(f"Courts FAISS index saved to {save_path}.faiss and .pkl")

    finally:
        # Ensure pool is stopped and memory released
        embedder.stop_multi_process_pool()


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
