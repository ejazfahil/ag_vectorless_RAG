"""
String-match evaluation metrics — F1, Exact Match, Precision, Recall.

Per blueprint C.3: direct span comparison without LLM judge.
For LegalBench-RAG follow the official precision/recall@k script.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> dict[str, float]:
    """
    Compute token-level F1, Precision, and Recall.

    Returns dict with keys: f1, precision, recall
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"f1": f1, "precision": precision, "recall": recall}


def compute_metrics_batch(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Compute aggregate metrics over a batch of predictions.

    Returns:
        Dict with mean F1, mean EM, mean Precision, mean Recall
    """
    assert len(predictions) == len(ground_truths), "Mismatched lengths"

    em_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for pred, gt in zip(predictions, ground_truths):
        em_scores.append(exact_match(pred, gt))
        metrics = token_f1(pred, gt)
        f1_scores.append(metrics["f1"])
        precision_scores.append(metrics["precision"])
        recall_scores.append(metrics["recall"])

    n = len(predictions)
    return {
        "exact_match": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "precision": sum(precision_scores) / n,
        "recall": sum(recall_scores) / n,
        "n": n,
    }
