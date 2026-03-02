#!/usr/bin/env python3
import argparse
import logging
import sys
import json
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_retrieval import FAISSIndex, MultilingualEmbedder
from omnilex.retrieval.translator import QueryTranslator
from omnilex.retrieval.citation_graph import CitationCooccurrenceGraph
from omnilex.retrieval.reranker import CrossEncoderReranker
from omnilex.pipeline.hybrid_retriever import HybridRetriever
from omnilex.pipeline.full_pipeline import FullPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tune_pipeline(args):
    # 1. Load data
    logger.info("Loading validation data...")
    val_df = pd.read_csv(args.val_csv)
    # Predicted citations in val.csv are semicolon separated canonical strings
    val_queries = val_df["query"].tolist()
    val_gold = [
        c.split(";") if pd.notna(c) else [] for c in val_df["gold_citations"].tolist()
    ]

    # 2. Load indices and models
    logger.info("Loading indices and models...")
    laws_bm25 = BM25Index.load(args.laws_bm25)
    courts_bm25 = BM25Index.load(args.courts_bm25)
    laws_faiss = FAISSIndex.load(args.laws_faiss)
    courts_faiss = FAISSIndex.load(args.courts_faiss)

    embedder = MultilingualEmbedder(model_name=args.embedder_model)
    translator = QueryTranslator(model_name=args.translator_model)
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    citation_graph = CitationCooccurrenceGraph.load(args.graph_path)

    # Build hard grounding set from corpora
    logger.info("Building corpus citation set...")
    laws_df = pd.read_csv(args.laws_csv)
    # Courts is too big for full load, use unique from FAISS pkl
    courts_citations = [d["citation"] for d in courts_faiss.documents]
    corpus_citation_set = set(laws_df["citation"].tolist()) | set(courts_citations)

    # 3. Initialize Pipeline
    hybrid_retriever = HybridRetriever(
        laws_bm25,
        courts_bm25,
        laws_faiss,
        courts_faiss,
        embedder,
        translator,
        citation_graph,
        corpus_citation_set,
    )

    pipeline = FullPipeline(
        hybrid_retriever, reranker, corpus_citation_set=corpus_citation_set
    )

    # 4. Tune
    best_threshold = pipeline.tune_threshold(val_queries, val_gold)

    # 5. Save config
    config_path = Path(args.output_config)
    pipeline.save_config(config_path)
    logger.info(f"Pipeline config with tuned threshold saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune pipeline threshold on validation set"
    )
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--laws-csv", type=str, required=True)
    parser.add_argument("--laws-bm25", type=str, required=True)
    parser.add_argument("--courts-bm25", type=str, required=True)
    parser.add_argument("--laws-faiss", type=str, required=True)
    parser.add_argument("--courts-faiss", type=str, required=True)
    parser.add_argument("--graph-path", type=str, required=True)
    parser.add_argument(
        "--output-config", type=str, default="config/pipeline_config.json"
    )

    # Model names
    parser.add_argument(
        "--embedder-model", type=str, default="intfloat/multilingual-e5-large"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    parser.add_argument(
        "--translator-model", type=str, default="Helsinki-NLP/opus-mt-en-de"
    )

    args = parser.parse_args()
    tune_pipeline(args)


if __name__ == "__main__":
    main()
