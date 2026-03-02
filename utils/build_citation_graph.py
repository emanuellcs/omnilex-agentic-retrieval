#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.citation_graph import CitationCooccurrenceGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_graph(args):
    graph = CitationCooccurrenceGraph()

    # Large file, process in chunks if not using max_rows
    if args.max_rows:
        logger.info(f"Loading first {args.max_rows} rows from {args.courts_csv}...")
        df = pd.read_csv(args.courts_csv, nrows=args.max_rows)
        graph.build_from_corpus(df)
    else:
        logger.info(f"Loading full corpus from {args.courts_csv} in chunks...")
        chunksize = 100000
        reader = pd.read_csv(args.courts_csv, chunksize=chunksize)
        for chunk in reader:
            graph.build_from_corpus(chunk)
            logger.info(
                f"Graph currently has {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges"
            )

    output_path = Path(args.output_path)
    graph.save(output_path)
    logger.info(f"Citation graph saved to {output_path}")

    # Print some stats
    logger.info(f"Total nodes: {graph.graph.number_of_nodes()}")
    logger.info(f"Total edges: {graph.graph.number_of_edges()}")

    # Top connected nodes
    degrees = sorted(graph.graph.degree(), key=lambda x: x[1], reverse=True)
    logger.info("Top-10 most connected citations:")
    for cit, deg in degrees[:10]:
        logger.info(f"  {cit}: {deg} connections")


def main():
    parser = argparse.ArgumentParser(description="Build citation co-occurrence graph")
    parser.add_argument(
        "--courts-csv", type=str, required=True, help="Path to court_considerations.csv"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/citation_graph.pkl",
        help="Output path",
    )
    parser.add_argument("--max-rows", type=int, help="Limit rows for testing")

    args = parser.parse_args()
    build_graph(args)


if __name__ == "__main__":
    main()
