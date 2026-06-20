"""Per-field accuracy and claim_status precision/recall/F1 against sample_claims.csv labels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import LIST_SEPARATOR  # noqa: E402

# Fields with a known expected value in sample_claims.csv that we score per-field accuracy on.
ACCURACY_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "severity",
    "valid_image",
    "issue_type",
    "object_part",
]

CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]

# Multi-value fields where token order shouldn't affect the accuracy comparison.
MULTI_VALUE_FIELDS = {"risk_flags", "supporting_image_ids"}


def _normalize(field: str, value: object) -> str:
    text = str(value).strip().lower()
    if field in MULTI_VALUE_FIELDS:
        tokens = sorted(t.strip() for t in text.split(LIST_SEPARATOR) if t.strip())
        return LIST_SEPARATOR.join(tokens)
    return text


def field_accuracy(predicted: list[dict], expected: list[dict]) -> dict[str, float]:
    n = len(predicted)
    accuracy = {}
    for field in ACCURACY_FIELDS:
        correct = sum(
            1
            for p, e in zip(predicted, expected)
            if _normalize(field, p.get(field)) == _normalize(field, e.get(field))
        )
        accuracy[field] = correct / n if n else 0.0
    return accuracy


def field_correct_wrong_counts(predicted: list[dict], expected: list[dict]) -> dict[str, dict[str, int]]:
    """Per-field correct/wrong row counts, for the summary-counts section."""
    n = len(predicted)
    counts = {}
    for field in ACCURACY_FIELDS:
        correct = sum(
            1
            for p, e in zip(predicted, expected)
            if _normalize(field, p.get(field)) == _normalize(field, e.get(field))
        )
        counts[field] = {"correct": correct, "wrong": n - correct}
    return counts


def claim_status_f1(predicted: list[dict], expected: list[dict]) -> dict:
    """Per-class precision/recall/F1 for claim_status, plus macro F1."""
    pred_labels = [_normalize("claim_status", p.get("claim_status")) for p in predicted]
    true_labels = [_normalize("claim_status", e.get("claim_status")) for e in expected]

    per_class = {}
    f1_scores = []
    for label in CLAIM_STATUS_LABELS:
        tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t == label)
        fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t != label)
        fn = sum(1 for p, t in zip(pred_labels, true_labels) if p != label and t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {"per_class": per_class, "macro_f1": macro_f1}


def claim_status_confusion_matrix(predicted: list[dict], expected: list[dict]) -> dict[str, dict[str, int]]:
    """3x3 confusion matrix over the fixed claim_status labels: matrix[true_label][pred_label]."""
    pred_labels = [_normalize("claim_status", p.get("claim_status")) for p in predicted]
    true_labels = [_normalize("claim_status", e.get("claim_status")) for e in expected]
    matrix = {t: {p: 0 for p in CLAIM_STATUS_LABELS} for t in CLAIM_STATUS_LABELS}
    for p, t in zip(pred_labels, true_labels):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def risk_flag_precision(predicted: list[dict], expected: list[dict]) -> float:
    """Micro-averaged precision of predicted risk_flags tokens, over rows where the
    expected output actually has risk_flags (i.e. not "none"/empty)."""
    intersection_total = 0
    predicted_total = 0
    for p, e in zip(predicted, expected):
        expected_tokens = set(_split(e.get("risk_flags")))
        if not expected_tokens or expected_tokens == {"none"}:
            continue
        predicted_tokens = set(_split(p.get("risk_flags")))
        predicted_total += len(predicted_tokens)
        intersection_total += len(predicted_tokens & expected_tokens)
    return intersection_total / predicted_total if predicted_total else 0.0


def _split(value: object) -> list[str]:
    text = str(value).strip().lower()
    return [t.strip() for t in text.split(LIST_SEPARATOR) if t.strip()]


def wrong_claim_status_rows(predicted: list[dict], expected: list[dict]) -> list[dict]:
    """Rows where claim_status was wrong, for the report's error-inspection table."""
    rows = []
    for p, e in zip(predicted, expected):
        pred_status = _normalize("claim_status", p.get("claim_status"))
        true_status = _normalize("claim_status", e.get("claim_status"))
        if pred_status != true_status:
            rows.append(
                {
                    "user_claim": e.get("user_claim", ""),
                    "predicted": pred_status,
                    "expected": true_status,
                }
            )
    return rows


def compute_metrics(predicted: list[dict], expected: list[dict]) -> dict:
    return {
        "n": len(predicted),
        "field_accuracy": field_accuracy(predicted, expected),
        "field_correct_wrong_counts": field_correct_wrong_counts(predicted, expected),
        "claim_status": claim_status_f1(predicted, expected),
        "claim_status_confusion_matrix": claim_status_confusion_matrix(predicted, expected),
        "risk_flag_precision": risk_flag_precision(predicted, expected),
        "wrong_claim_status_rows": wrong_claim_status_rows(predicted, expected),
    }
