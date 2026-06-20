"""Evaluation entry point.

Usage:
    python code/evaluation/main.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncio
import pandas as pd
from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from image_utils import split_image_paths  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from pipeline import load_evidence_requirements, load_user_history, run_pipeline  # noqa: E402
from report_utils import format_operational_section  # noqa: E402
from schemas import OUTPUT_COLUMNS  # noqa: E402

REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_PATH = DATASET_DIR / "sample_claims.csv"
EVAL_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = EVAL_DIR / "sample_predictions.csv"
REPORT_PATH = EVAL_DIR / "evaluation_report.md"

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]


async def run() -> None:
    load_dotenv()
    sample_df = pd.read_csv(SAMPLE_CLAIMS_PATH, dtype=str).fillna("")
    claims = sample_df[INPUT_COLUMNS].to_dict("records")
    expected = sample_df.to_dict("records")

    history_by_user = load_user_history(DATASET_DIR / "user_history.csv")
    evidence_requirements = load_evidence_requirements(DATASET_DIR / "evidence_requirements.csv")

    results, stats = await run_pipeline(claims, history_by_user, evidence_requirements, DATASET_DIR)

    predicted = [r.as_row() for r in results]
    predictions_df = pd.DataFrame(predicted, columns=OUTPUT_COLUMNS)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False, quoting=csv.QUOTE_ALL)

    metrics = compute_metrics(predicted, expected)
    images_processed = sum(len(split_image_paths(c["image_paths"])) for c in claims)
    stats.images_processed = images_processed

    write_report(metrics, stats)
    print(f"Wrote predictions to {PREDICTIONS_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


def write_report(metrics: dict, stats) -> None:
    n = metrics["n"]
    run_time = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "# Evaluation Report",
        "",
        "## ACCURACY (dataset/sample_claims.csv only)",
        "",
        f"Run at: {run_time}",
        f"Evaluated on {n} rows from `dataset/sample_claims.csv`.",
        "",
        "### Per-field accuracy",
        "",
        "| field | accuracy | correct | wrong |",
        "|---|---|---|---|",
    ]
    for field, value in metrics["field_accuracy"].items():
        counts = metrics["field_correct_wrong_counts"][field]
        lines.append(f"| `{field}` | {value:.1%} | {counts['correct']} | {counts['wrong']} |")

    lines += [
        "",
        "### claim_status confusion matrix",
        "",
        "| true \\ predicted | supported | contradicted | not_enough_information |",
        "|---|---|---|---|",
    ]
    matrix = metrics["claim_status_confusion_matrix"]
    for true_label in ["supported", "contradicted", "not_enough_information"]:
        row = matrix[true_label]
        lines.append(
            f"| {true_label} | {row['supported']} | {row['contradicted']} | "
            f"{row['not_enough_information']} |"
        )

    lines += [
        "",
        "### claim_status precision / recall / F1",
        "",
        "| label | precision | recall | f1 |",
        "|---|---|---|---|",
    ]
    for label, scores in metrics["claim_status"]["per_class"].items():
        lines.append(
            f"| {label} | {scores['precision']:.2f} | {scores['recall']:.2f} | {scores['f1']:.2f} |"
        )
    lines.append("")
    lines.append(f"Macro F1: {metrics['claim_status']['macro_f1']:.2f}")

    lines += [
        "",
        f"### Risk flag precision: {metrics['risk_flag_precision']:.1%}",
        "",
        "Fraction of predicted risk_flags tokens that appear in the expected risk_flags, "
        "over rows where the expected output actually has risk_flags.",
        "",
        "### Rows where claim_status was wrong",
        "",
    ]
    wrong_rows = metrics["wrong_claim_status_rows"]
    if not wrong_rows:
        lines.append("None -- claim_status matched on every row.")
    else:
        lines += [
            "| user_claim | predicted | expected |",
            "|---|---|---|",
        ]
        for row in wrong_rows:
            user_claim = row["user_claim"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {user_claim} | {row['predicted']} | {row['expected']} |")
    lines.append("")

    lines += format_operational_section("SAMPLE OPERATIONAL ANALYSIS (dataset/sample_claims.csv)", stats, n)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
