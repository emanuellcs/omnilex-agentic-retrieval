"""Evaluation metrics for citation retrieval.

Primary metric: Macro F1 (average F1 across all queries)
Secondary metrics: Precision, Recall, MAP, NDCG
"""

from collections.abc import Sequence


def citation_f1(
    predicted: Sequence[str],
    gold: Sequence[str],
) -> dict[str, float]:
    """Compute F1 score for citation overlap on a single query.

    Args:
        predicted: List of predicted canonical citation IDs
        gold: List of ground truth canonical citation IDs

    Returns:
        Dictionary with precision, recall, and F1
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    # Edge case: both empty
    if len(pred_set) == 0 and len(gold_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Edge case: prediction empty but gold not
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Edge case: gold empty but prediction not
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Macro F1: average F1 across all queries.

    This is the PRIMARY competition metric.

    Args:
        predictions: List of predicted citation lists (one per query)
        gold: List of gold citation lists (one per query)

    Returns:
        Dictionary with macro precision, recall, and F1
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold"
        )

    if len(predictions) == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, g in zip(predictions, gold):
        scores = citation_f1(pred, g)
        precision_scores.append(scores["precision"])
        recall_scores.append(scores["recall"])
        f1_scores.append(scores["f1"])

    n = len(f1_scores)
    return {
        "macro_precision": sum(precision_scores) / n,
        "macro_recall": sum(recall_scores) / n,
        "macro_f1": sum(f1_scores) / n,
    }


def micro_f1(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Micro F1: aggregate TP/FP/FN across all queries.

    Args:
        predictions: List of predicted citation lists (one per query)
        gold: List of gold citation lists (one per query)

    Returns:
        Dictionary with micro precision, recall, and F1
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold"
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, g in zip(predictions, gold):
        pred_set = set(pred)
        gold_set = set(g)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def average_precision(
    predicted_ranked: Sequence[str],
    gold: Sequence[str],
) -> float:
    """Compute Average Precision for a single query.

    Args:
        predicted_ranked: Ranked list of predicted citations (most relevant first)
        gold: Set of gold citations (order doesn't matter)

    Returns:
        Average Precision score
    """
    gold_set = set(gold)

    if not gold_set:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, pred in enumerate(predicted_ranked, 1):
        if pred in gold_set:
            hits += 1
            precision_sum += hits / i

    return precision_sum / len(gold_set)


def mean_average_precision(
    predictions_ranked: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> float:
    """Compute Mean Average Precision (MAP) across all queries.

    Use this metric when citation ranking matters (not just set overlap).

    Args:
        predictions_ranked: List of ranked predictions per query
        gold: List of gold citations per query

    Returns:
        Mean Average Precision score
    """
    if len(predictions_ranked) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions_ranked)} predictions vs {len(gold)} gold"
        )

    if len(predictions_ranked) == 0:
        return 0.0

    ap_scores = []
    for pred, g in zip(predictions_ranked, gold):
        ap = average_precision(pred, g)
        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores)


def ndcg_at_k(
    predicted_ranked: Sequence[str],
    gold: Sequence[str],
    k: int = 10,
) -> float:
    """Compute NDCG@k for a single query.

    Uses binary relevance (1 if in gold set, 0 otherwise).

    Args:
        predicted_ranked: Ranked list of predicted citations
        gold: Set of gold citations
        k: Cutoff for evaluation

    Returns:
        NDCG@k score
    """
    import math

    gold_set = set(gold)

    if not gold_set:
        return 0.0

    # DCG
    dcg = 0.0
    for i, pred in enumerate(predicted_ranked[:k], 1):
        rel = 1.0 if pred in gold_set else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG (all gold items ranked first)
    idcg = 0.0
    for i in range(1, min(len(gold_set), k) + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_ndcg_at_k(
    predictions_ranked: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    k: int = 10,
) -> float:
    """Compute Mean NDCG@k across all queries.

    Args:
        predictions_ranked: List of ranked predictions per query
        gold: List of gold citations per query
        k: Cutoff for evaluation

    Returns:
        Mean NDCG@k score
    """
    if len(predictions_ranked) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions_ranked)} predictions vs {len(gold)} gold"
        )

    if len(predictions_ranked) == 0:
        return 0.0

    ndcg_scores = []
    for pred, g in zip(predictions_ranked, gold):
        ndcg = ndcg_at_k(pred, g, k)
        ndcg_scores.append(ndcg)

    return sum(ndcg_scores) / len(ndcg_scores)
