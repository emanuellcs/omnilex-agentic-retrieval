#!/usr/bin/env python3
import argparse
import gc
import logging
import os
import subprocess
import sys
from pathlib import Path

# Avoid CUDA context creation during torch.cuda availability checks in the parent.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

# Ensure cuDF is disabled to prevent VRAM hoarding
try:
    if "cudf.pandas" in sys.modules or "cudf" in sys.modules:
        import cudf.pandas

        cudf.pandas.uninstall()
        print("Explicitly uninstalled cudf.pandas")
except Exception:
    pass


def print_vram_usage(step: str):
    """Log process-level GPU memory without initializing CUDA in this process."""
    gpu_rows = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_rows is None:
        print(f"TELEMETRY: {step}: nvidia-smi unavailable")
        return

    process_rows = _run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    uuid_to_index = {}
    gpu_summaries = []
    for row in gpu_rows:
        if len(row) < 6:
            continue
        index, uuid, name, used, free, total = row[:6]
        uuid_to_index[uuid] = index
        gpu_summaries.append(
            f"GPU {index} {name}: used={used} MiB free={free} MiB total={total} MiB"
        )

    processes_by_gpu: dict[str, list[str]] = {}
    for row in process_rows or []:
        if len(row) < 4:
            continue
        gpu_uuid, pid, process_name, used = row[:4]
        gpu_index = uuid_to_index.get(gpu_uuid, gpu_uuid)
        processes_by_gpu.setdefault(gpu_index, []).append(
            f"pid={pid} used={used} MiB name={process_name}"
        )

    print(f"TELEMETRY: {step}")
    for summary in gpu_summaries:
        print(f"  {summary}")
    if processes_by_gpu:
        for gpu_index, processes in sorted(processes_by_gpu.items()):
            print(f"  GPU {gpu_index} processes: {'; '.join(processes)}")
    else:
        print("  No active GPU compute processes reported by nvidia-smi.")


def _run_nvidia_smi(args: list[str]) -> list[list[str]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return []

    rows = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("No running processes"):
            continue
        rows.append([part.strip() for part in stripped.split(",")])
    return rows


def _parse_devices(value: str | None) -> list[str] | None:
    if not value:
        return None
    devices = [device.strip() for device in value.split(",") if device.strip()]
    return devices or None


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indices(args):
    import pandas as pd
    from tqdm import tqdm

    from omnilex.retrieval.dense_retrieval import FAISSIndex, MultilingualEmbedder

    print_vram_usage("build_indices start")
    devices = _parse_devices(args.devices)
    embedder = MultilingualEmbedder(
        model_name=args.model_name,
        devices=devices,
        batch_size=args.batch_size,
        chunk_size=args.encode_chunk_size,
        dtype=args.embedding_dtype,
        max_seq_length=args.max_seq_length,
    )
    print_vram_usage("MultilingualEmbedder initialized")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start persistent pool for multiple GPUs
    embedder.start_multi_process_pool()
    print_vram_usage("Embedding worker pool started")

    try:
        # 1. Process Laws
        if args.laws_csv:
            logger.info(f"Processing laws from {args.laws_csv}")
            df_laws = pd.read_csv(args.laws_csv)
            print_vram_usage("Laws CSV loaded")
            if args.max_rows_laws:
                df_laws = df_laws.head(args.max_rows_laws)

            texts = df_laws["text"].tolist()
            logger.info(f"Encoding {len(texts)} law passages...")
            embeddings = embedder.encode(texts, is_query=False)

            index = FAISSIndex()
            print_vram_usage("Laws FAISSIndex initialized")
            docs = df_laws[["citation", "text"]].to_dict("records")

            logger.info("Building FAISS index for laws...")
            index.build(embeddings, docs)

            save_path = output_dir / "laws_faiss"
            index.save(save_path)
            logger.info(f"Laws FAISS index saved to {save_path}.faiss and .pkl")

            # Clean up aggressively
            del embeddings, docs, df_laws, texts
            gc.collect()

        # 2. Process Courts (Chunked & Incremental)
        if args.courts_csv:
            logger.info(f"Processing courts from {args.courts_csv}")

            index = FAISSIndex()
            print_vram_usage("Courts FAISSIndex initialized")
            is_trained = False

            # Load in chunks to avoid RAM OOM
            reader = pd.read_csv(args.courts_csv, chunksize=args.csv_chunksize)
            print_vram_usage("Courts CSV reader initialized")

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
                    "Encoding chunk of %s court passages (Total processed: %s)...",
                    len(texts),
                    row_count,
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
                print_vram_usage(f"court chunk {row_count} processed")

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
    parser.add_argument(
        "--devices",
        type=str,
        help="Comma-separated embedding devices, e.g. cuda:0,cuda:1",
    )
    parser.add_argument(
        "--encode-chunk-size",
        type=int,
        help="Number of texts sent to each embedding worker task",
    )
    parser.add_argument(
        "--embedding-dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "fp16", "float32", "fp32"],
        help="Embedding model dtype. CUDA defaults should use float16 on T4.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Optional sentence-transformers max sequence length override",
    )
    parser.add_argument(
        "--csv-chunksize",
        type=int,
        default=50000,
        help="Rows per Pandas CSV chunk for court considerations",
    )
    parser.add_argument("--max-rows-laws", type=int, help="Limit laws rows for testing")
    parser.add_argument(
        "--max-rows-courts", type=int, help="Limit courts rows for testing"
    )

    args = parser.parse_args()
    build_indices(args)


if __name__ == "__main__":
    main()
