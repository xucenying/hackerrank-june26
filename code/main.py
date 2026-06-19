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

from pipeline import load_evidence_requirements, load_user_history, run_pipeline
from schemas import OUTPUT_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"


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

    results = await run_pipeline(claims, history_by_user, evidence_requirements, args.dataset_dir)

    output_df = pd.DataFrame([result.as_row() for result in results], columns=OUTPUT_COLUMNS)
    output_df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(output_df)} rows to {args.output}")


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
