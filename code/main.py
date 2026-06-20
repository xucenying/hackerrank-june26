"""Terminal entry point.

Usage:
    python code/main.py --input dataset/claims.csv --output output.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from image_utils import split_image_paths
from pipeline import load_evidence_requirements, load_user_history, run_pipeline
from report_utils import format_operational_section
from schemas import OUTPUT_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
EVAL_REPORT_PATH = REPO_ROOT / "code" / "evaluation" / "evaluation_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-modal evidence review pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET_DIR / "claims.csv",
        help="Path to the claims CSV to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output.csv",
        help="Path to write the output CSV",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory containing user_history.csv, evidence_requirements.csv, and images/",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    claims_df = pd.read_csv(args.input, dtype=str).fillna("")
    claims = claims_df.to_dict("records")

    history_by_user = load_user_history(args.dataset_dir / "user_history.csv")
    evidence_requirements = load_evidence_requirements(args.dataset_dir / "evidence_requirements.csv")

    results, stats = await run_pipeline(claims, history_by_user, evidence_requirements, args.dataset_dir)

    output_df = pd.DataFrame([result.as_row() for result in results], columns=OUTPUT_COLUMNS)
    output_df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(output_df)} rows to {args.output}")

    stats.images_processed = sum(len(split_image_paths(c["image_paths"])) for c in claims)
    append_test_operational_analysis(stats, len(claims))
    print(f"Appended TEST OPERATIONAL ANALYSIS section to {EVAL_REPORT_PATH}")


def append_test_operational_analysis(stats, row_count: int) -> None:
    """Append a TEST OPERATIONAL ANALYSIS section (no accuracy -- claims.csv has no
    expected outputs) to evaluation_report.md. Each run on claims.csv adds a new
    section rather than replacing prior ones, so the report keeps a history of runs."""
    section = format_operational_section("TEST OPERATIONAL ANALYSIS (dataset/claims.csv)", stats, row_count)
    existing = EVAL_REPORT_PATH.read_text(encoding="utf-8") if EVAL_REPORT_PATH.exists() else "# Evaluation Report\n"
    EVAL_REPORT_PATH.write_text(existing.rstrip("\n") + "\n\n" + "\n".join(section), encoding="utf-8")


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
