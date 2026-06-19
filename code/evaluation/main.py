"""Evaluation entry point.

Usage:
    python code/evaluation/main.py
"""

from __future__ import annotations

import asyncio
import csv
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import vlm_client  # noqa: E402
from image_utils import split_image_paths  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from pipeline import load_evidence_requirements, load_user_history, run_pipeline  # noqa: E402
from schemas import OUTPUT_COLUMNS  # noqa: E402

REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_PATH = DATASET_DIR / "sample_claims.csv"
TEST_CLAIMS_PATH = DATASET_DIR / "claims.csv"
EVAL_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = EVAL_DIR / "sample_predictions.csv"
REPORT_PATH = EVAL_DIR / "evaluation_report.md"

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]

# Pricing assumption for the operational-analysis section (per problem_statement.md
# "Operational analysis"). Claude Sonnet 4.6: $3.00 / MTok input, $15.00 / MTok output.
INPUT_PRICE_PER_MTOK = 3.00
OUTPUT_PRICE_PER_MTOK = 15.00


async def run() -> None:
    load_dotenv()
    sample_df = pd.read_csv(SAMPLE_CLAIMS_PATH, dtype=str).fillna("")
    claims = sample_df[INPUT_COLUMNS].to_dict("records")
    expected = sample_df.to_dict("records")

    history_by_user = load_user_history(DATASET_DIR / "user_history.csv")
    evidence_requirements = load_evidence_requirements(DATASET_DIR / "evidence_requirements.csv")

    vlm_client.reset_usage_stats()
    start = time.monotonic()
    results = await run_pipeline(claims, history_by_user, evidence_requirements, DATASET_DIR)
    elapsed_seconds = time.monotonic() - start
    usage = vlm_client.get_usage_stats()

    predicted = [r.as_row() for r in results]
    predictions_df = pd.DataFrame(predicted, columns=OUTPUT_COLUMNS)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False, quoting=csv.QUOTE_ALL)

    metrics = compute_metrics(predicted, expected)
    images_processed = sum(len(split_image_paths(c["image_paths"])) for c in claims)
    test_claim_count = _count_test_claims()

    write_report(metrics, usage, elapsed_seconds, images_processed, test_claim_count)
    print(f"Wrote predictions to {PREDICTIONS_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


def _count_test_claims() -> int:
    if not TEST_CLAIMS_PATH.exists():
        return 0
    return len(pd.read_csv(TEST_CLAIMS_PATH, dtype=str))


def write_report(
    metrics: dict,
    usage: dict,
    elapsed_seconds: float,
    images_processed: int,
    test_claim_count: int,
) -> None:
    n = metrics["n"]
    calls = usage["call_count"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    extra_fix_calls = max(calls - n, 0)
    cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK + (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MTOK
    avg_calls_per_claim = calls / n if n else 0.0
    avg_input_tokens = input_tokens / calls if calls else 0.0
    avg_output_tokens = output_tokens / calls if calls else 0.0
    projected_test_calls = avg_calls_per_claim * test_claim_count
    projected_test_input_tokens = avg_input_tokens * projected_test_calls
    projected_test_output_tokens = avg_output_tokens * projected_test_calls
    projected_test_cost = (
        (projected_test_input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK
        + (projected_test_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MTOK
    )
    projected_test_seconds = (elapsed_seconds / n) * test_claim_count if n else 0.0

    lines = [
        "# Evaluation Report",
        "",
        f"Evaluated on {n} rows from `dataset/sample_claims.csv`.",
        "",
        "## Per-field accuracy",
        "",
    ]
    for field, value in metrics["field_accuracy"].items():
        lines.append(f"- `{field}`: {value:.1%}")

    lines += [
        "",
        "## claim_status precision / recall / F1",
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
        "## Operational analysis",
        "",
        f"- Model calls (sample run, {n} claims): {calls} "
        f"({n} Stage 2 calls + {extra_fix_calls} Stage 3a enum-fix retries)",
        f"- Images processed (sample run): {images_processed}",
        f"- Token usage (sample run): {input_tokens:,} input / {output_tokens:,} output "
        f"(avg {avg_input_tokens:,.0f} in / {avg_output_tokens:,.0f} out per call)",
        f"- Measured cost (sample run): ${cost:.4f} "
        f"(assuming Claude Sonnet 4.6 pricing: ${INPUT_PRICE_PER_MTOK:.2f}/MTok input, "
        f"${OUTPUT_PRICE_PER_MTOK:.2f}/MTok output)",
        f"- Runtime (sample run): {elapsed_seconds:.1f}s wall clock under asyncio.Semaphore(5) concurrency",
        "",
        f"### Projected for the full test set ({test_claim_count} claims in `dataset/claims.csv`)",
        "",
        f"- Projected model calls: ~{projected_test_calls:.0f} "
        f"(linear scale-up of the sample's {avg_calls_per_claim:.2f} calls/claim, "
        "including expected enum-fix retries)",
        f"- Projected cost: ~${projected_test_cost:.2f}",
        f"- Projected runtime: ~{projected_test_seconds:.0f}s wall clock at the same concurrency",
        "",
        "### Rate limits, batching, and retry strategy",
        "",
        "- Concurrency is capped at `asyncio.Semaphore(5)` (`pipeline.CONCURRENCY_LIMIT`) to stay "
        "well under per-minute request and token limits for a single API key; raise this only "
        "after checking the account's actual RPM/TPM tier.",
        "- `vlm_client.call_vlm` retries only on HTTP 429 (rate limited) and 529 (overloaded) with "
        "exponential backoff plus jitter, max 3 attempts; other errors fail fast rather than masking "
        "a real problem behind retries.",
        "- No request-level caching or batching is implemented in this first benchmark pass -- each "
        "claim is one independent Stage 2 call. Since the system prompt is identical across every "
        "claim, adding `cache_control` to it would cut input-token cost on every call after the "
        "first; the Batches API (50% cost) would also fit well here since this is an offline, "
        "non-interactive job with no per-claim latency requirement.",
        "- Stage 3a's enum-fix retry is bounded to exactly one extra call per invalid field, so a "
        "single malformed claim cannot cascade into unbounded extra spend.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
