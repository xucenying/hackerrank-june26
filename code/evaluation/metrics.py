"""Per-field accuracy and claim_status precision/recall/F1 against sample_claims.csv labels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import LIST_SEPARATOR, OUTPUT_COLUMNS  # noqa: E402

# Output fields the model is scored on (excludes the 4 passthrough input fields).
SCORED_FIELDS = OUTPUT_COLUMNS[4:]

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
    for field in SCORED_FIELDS:
        correct = sum(
            1
            for p, e in zip(predicted, expected)
            if _normalize(field, p.get(field)) == _normalize(field, e.get(field))
        )
        accuracy[field] = correct / n if n else 0.0
    return accuracy


def claim_status_f1(predicted: list[dict], expected: list[dict]) -> dict:
    """Per-class precision/recall/F1 for claim_status, plus macro F1."""
    pred_labels = [_normalize("claim_status", p.get("claim_status")) for p in predicted]
    true_labels = [_normalize("claim_status", e.get("claim_status")) for e in expected]
    labels = sorted(set(true_labels) | set(pred_labels))

    per_class = {}
    f1_scores = []
    for label in labels:
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


def compute_metrics(predicted: list[dict], expected: list[dict]) -> dict:
    return {
        "n": len(predicted),
        "field_accuracy": field_accuracy(predicted, expected),
        "claim_status": claim_status_f1(predicted, expected),
    }
