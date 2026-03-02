"""Submission scoring and validation."""

from pathlib import Path

import pandas as pd

from ..citations.normalizer import CitationNormalizer
from .metrics import macro_f1, mean_average_precision, micro_f1


class Scorer:
    """Score competition submissions against gold standard.

    Handles submission validation, citation normalization, and metric computation.
    """

    def __init__(
        self,
        normalizer: CitationNormalizer | None = None,
        citation_separator: str = ";",
    ):
        """Initialize scorer.

        Args:
            normalizer: Citation normalizer instance (creates default if None)
            citation_separator: Separator used between citations in submission
        """
        self.normalizer = normalizer or CitationNormalizer()
        self.citation_separator = citation_separator

    def load_submission(self, submission_path: Path | str) -> pd.DataFrame:
        """Load and validate a submission file.

        Args:
            submission_path: Path to submission CSV

        Returns:
            DataFrame with query_id and predicted_citations columns

        Raises:
            ValueError: If submission format is invalid
        """
        submission_path = Path(submission_path)

        if not submission_path.exists():
            raise ValueError(f"Submission file not found: {submission_path}")

        df = pd.read_csv(submission_path)

        # Validate required columns
        required_cols = {"query_id", "predicted_citations"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Fill NaN with empty string
        df["predicted_citations"] = df["predicted_citations"].fillna("")

        return df

    def load_gold(self, gold_path: Path | str) -> pd.DataFrame:
        """Load gold standard file.

        Args:
            gold_path: Path to gold CSV

        Returns:
            DataFrame with query_id and gold_citations columns
        """
        gold_path = Path(gold_path)

        if not gold_path.exists():
            raise ValueError(f"Gold file not found: {gold_path}")

        df = pd.read_csv(gold_path)

        # Validate required columns
        required_cols = {"query_id", "gold_citations"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["gold_citations"] = df["gold_citations"].fillna("")

        return df

    def parse_citations(self, citation_string: str) -> list[str]:
        """Parse citation string into normalized list.

        Args:
            citation_string: Semicolon-separated citation string

        Returns:
            List of normalized canonical citation IDs
        """
        if not citation_string or citation_string.strip() == "":
            return []

        raw_citations = [
            c.strip()
            for c in citation_string.split(self.citation_separator)
            if c.strip()
        ]

        return self.normalizer.canonicalize_list(raw_citations)

    def score(
        self,
        submission_path: Path | str,
        gold_path: Path | str,
    ) -> dict[str, float]:
        """Score a submission against gold standard.

        Args:
            submission_path: Path to submission CSV
            gold_path: Path to gold CSV

        Returns:
            Dictionary with all metric scores
        """
        submission_df = self.load_submission(submission_path)
        gold_df = self.load_gold(gold_path)

        # Merge on query_id
        merged = pd.merge(
            submission_df,
            gold_df,
            on="query_id",
            how="outer",
            indicator=True,
        )

        # Check for missing queries
        missing_in_submission = merged[merged["_merge"] == "right_only"]["query_id"]
        if len(missing_in_submission) > 0:
            raise ValueError(
                f"Submission missing queries: {missing_in_submission.tolist()}"
            )

        extra_in_submission = merged[merged["_merge"] == "left_only"]["query_id"]
        if len(extra_in_submission) > 0:
            print(
                f"Warning: Submission has extra queries: {extra_in_submission.tolist()}"
            )

        # Filter to only matched queries
        merged = merged[merged["_merge"] == "both"]

        # Parse citations
        predictions = [
            self.parse_citations(row["predicted_citations"])
            for _, row in merged.iterrows()
        ]
        gold = [
            self.parse_citations(row["gold_citations"]) for _, row in merged.iterrows()
        ]

        # Compute metrics
        macro_scores = macro_f1(predictions, gold)
        micro_scores = micro_f1(predictions, gold)
        map_score = mean_average_precision(predictions, gold)

        return {
            **macro_scores,
            **micro_scores,
            "map": map_score,
            "num_queries": len(predictions),
        }


def evaluate_submission(
    submission_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a submission DataFrame against gold DataFrame.

    Convenience function for notebook evaluation.

    Args:
        submission_df: DataFrame with query_id and predicted_citations
        gold_df: DataFrame with query_id and gold_citations
        metrics: List of metrics to compute (default: all)

    Returns:
        Dictionary with requested metric scores
    """
    scorer = Scorer()

    # Merge DataFrames
    merged = pd.merge(
        submission_df,
        gold_df,
        on="query_id",
        how="inner",
    )

    # Parse citations
    predictions = [
        scorer.parse_citations(row.get("predicted_citations", ""))
        for _, row in merged.iterrows()
    ]
    gold = [
        scorer.parse_citations(row.get("gold_citations", ""))
        for _, row in merged.iterrows()
    ]

    # Compute all scores
    all_scores = {}

    macro_scores = macro_f1(predictions, gold)
    micro_scores = micro_f1(predictions, gold)
    map_score = mean_average_precision(predictions, gold)

    all_scores.update(macro_scores)
    all_scores.update(micro_scores)
    all_scores["map"] = map_score

    # Filter to requested metrics
    if metrics:
        metric_mapping = {
            "f1": "macro_f1",
            "precision": "macro_precision",
            "recall": "macro_recall",
            "macro_f1": "macro_f1",
            "micro_f1": "micro_f1",
            "map": "map",
        }
        filtered = {}
        for m in metrics:
            key = metric_mapping.get(m, m)
            if key in all_scores:
                filtered[m] = all_scores[key]
        return filtered

    return all_scores


def validate_submission_format(submission_path: Path | str) -> list[str]:
    """Validate submission file format without scoring.

    Args:
        submission_path: Path to submission CSV

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    submission_path = Path(submission_path)

    # Check file exists
    if not submission_path.exists():
        return [f"File not found: {submission_path}"]

    # Check file extension
    if submission_path.suffix.lower() != ".csv":
        errors.append(f"Expected .csv file, got {submission_path.suffix}")

    # Try to load
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        return [f"Failed to parse CSV: {e}"]

    # Check columns
    required_cols = {"query_id", "predicted_citations"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    # Check for empty query_ids
    if df["query_id"].isna().any():
        errors.append("Found NaN values in query_id column")

    # Check for duplicate query_ids
    duplicates = df[df["query_id"].duplicated()]["query_id"].tolist()
    if duplicates:
        errors.append(f"Found duplicate query_ids: {duplicates[:5]}...")

    # Validate citation format (sample check)
    normalizer = CitationNormalizer()
    sample_size = min(10, len(df))
    unparseable = []

    for _, row in df.head(sample_size).iterrows():
        citations_str = row.get("predicted_citations", "")
        if pd.isna(citations_str) or citations_str == "":
            continue

        for citation in citations_str.split(";"):
            citation = citation.strip()
            if citation and not normalizer.normalize(citation):
                unparseable.append(citation)

    if unparseable:
        errors.append(f"Found unparseable citations in sample: {unparseable[:3]}...")

    return errors
