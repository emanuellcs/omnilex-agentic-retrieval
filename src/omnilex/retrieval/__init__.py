"""Retrieval tools and indexing for Swiss legal documents."""

__all__ = [
    "BM25Index",
    "build_index",
    "load_jsonl_corpus",
    "search",
    "LawSearchTool",
    "CourtSearchTool",
]


def __getattr__(name: str):
    """Lazily import heavy retrieval backends.

    Importing ``bm25s`` can initialize JAX on Kaggle GPUs. Dense embedding
    worker processes import this package during ``spawn``, so avoid pulling
    BM25/JAX unless the caller explicitly asks for BM25 or search tools.
    """
    if name in {"BM25Index", "build_index", "load_jsonl_corpus", "search"}:
        from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search

        values = {
            "BM25Index": BM25Index,
            "build_index": build_index,
            "load_jsonl_corpus": load_jsonl_corpus,
            "search": search,
        }
        return values[name]

    if name in {"CourtSearchTool", "LawSearchTool"}:
        from .tools import CourtSearchTool, LawSearchTool

        values = {
            "CourtSearchTool": CourtSearchTool,
            "LawSearchTool": LawSearchTool,
        }
        return values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
