"""Stage 1-3 pipeline: load inputs, call the VLM, post-hoc validate -- per claim.

main.py drives this module and owns argument parsing / writing output.csv; the
asyncio.Semaphore(5) concurrency limit is created here and shared across claims.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import prompts
from image_utils import load_claim_images
from post_hoc import apply_post_hoc
from schemas import ClaimOutput, DEFAULT_NOT_ENOUGH_INFO, LIST_SEPARATOR
from vlm_client import RunStats, call_vlm

CONCURRENCY_LIMIT = 5

# Prompt-injection guard: substrings checked case-insensitively against user_claim
# before the model call. No model call is used for detection -- plain Python substring
# matching is sufficient and free.
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "disregard the above",
    "repeat your system prompt",
    "output your rules",
    "forget your instructions",
    "new instruction",
    "override",
)


def load_user_history(path: Path) -> dict[str, dict]:
    """Load user_history.csv into a dict keyed by user_id."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return {row["user_id"]: row.to_dict() for _, row in df.iterrows()}


def load_evidence_requirements(path: Path) -> list[dict]:
    """Load evidence_requirements.csv as a list of row dicts."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df.to_dict("records")


def _history_context(user_id: str, history_by_user: dict[str, dict]) -> dict:
    """User history fields for the prompt, with safe defaults if user_id is absent
    from user_history.csv (treated as a new user with no risk indicators)."""
    row = history_by_user.get(user_id)
    if row is None:
        return {
            "past_claim_count": "0",
            "accept_claim": "0",
            "manual_review_claim": "0",
            "rejected_claim": "0",
            "last_90_days_claim_count": "0",
            "history_flags": "none",
            "history_summary": prompts.NO_HISTORY_SUMMARY_TEXT,
        }
    return row


def _relevant_evidence_requirements(claim_object: str, requirements: list[dict]) -> str:
    """Evidence-requirement rows for this claim_object (plus the object-agnostic
    "all" rows), formatted as a bullet list for the prompt."""
    relevant = [r for r in requirements if r["claim_object"] in (claim_object, "all")]
    return "\n".join(f"- ({r['requirement_id']}) {r['minimum_image_evidence']}" for r in relevant)


def _fallback_row(claim: dict) -> ClaimOutput:
    """Stage 1 short-circuit: all submitted images were missing/unreadable."""
    return ClaimOutput(
        user_id=claim["user_id"],
        image_paths=claim["image_paths"],
        user_claim=claim["user_claim"],
        claim_object=claim["claim_object"],
        **DEFAULT_NOT_ENOUGH_INFO,
    )


def _sanitize_user_claim(user_claim: str) -> tuple[str, bool]:
    """Scan user_claim for known prompt-injection substrings (case-insensitive) and
    redact any match before the text reaches the model. Returns (sanitized_text,
    was_detected). The unsanitized original is still echoed in the output row's
    user_claim column -- only the model-facing copy is redacted."""
    detected = False
    sanitized = user_claim
    for pattern in INJECTION_PATTERNS:
        if pattern in sanitized.lower():
            detected = True
            sanitized = re.sub(re.escape(pattern), "[redacted]", sanitized, flags=re.IGNORECASE)
    return sanitized, detected


def _merge_risk_flag(risk_flags: str, flag: str) -> str:
    """Add flag to a semicolon-separated risk_flags string, deduplicating."""
    tokens = [] if risk_flags == "none" else risk_flags.split(LIST_SEPARATOR)
    if flag not in tokens:
        tokens.append(flag)
    return LIST_SEPARATOR.join(tokens)


async def process_claim(
    claim: dict,
    history_by_user: dict[str, dict],
    evidence_requirements: list[dict],
    images_base_dir: Path,
    semaphore: asyncio.Semaphore,
    stats: RunStats,
) -> ClaimOutput:
    """Run Stage 1-3 for a single claims.csv row."""
    image_result = load_claim_images(claim["image_paths"], images_base_dir)

    if not image_result.valid_images:
        return _fallback_row(claim)

    stats.images_processed += len(image_result.valid_images)
    history = _history_context(claim["user_id"], history_by_user)
    image_ids = ";".join(img.image_id for img in image_result.valid_images)
    sanitized_claim, injection_detected = _sanitize_user_claim(claim["user_claim"])

    user_text = prompts.CLAIM_ANALYSIS_PROMPT_TEMPLATE.format(
        claim_object=claim["claim_object"],
        user_claim=sanitized_claim,
        image_ids=image_ids,
        evidence_requirements=_relevant_evidence_requirements(
            claim["claim_object"], evidence_requirements
        ),
        past_claim_count=history["past_claim_count"],
        accept_claim=history["accept_claim"],
        manual_review_claim=history["manual_review_claim"],
        rejected_claim=history["rejected_claim"],
        last_90_days_claim_count=history["last_90_days_claim_count"],
        history_flags=history["history_flags"],
        history_summary=history["history_summary"],
        risk_flags=prompts.RISK_FLAGS_TEXT,
        issue_types=prompts.issue_types_text(claim["claim_object"]),
        object_parts=prompts.object_parts_text(claim["claim_object"]),
        claim_statuses=prompts.CLAIM_STATUSES_TEXT,
        severities=prompts.SEVERITIES_TEXT,
    )

    async with semaphore:
        model_output = await call_vlm(
            claim["claim_object"], user_text, image_result.valid_images, stats
        )

    result = await apply_post_hoc(
        claim=claim,
        history=history,
        model_output=model_output,
        valid_images=image_result.valid_images,
        invalid_paths=image_result.invalid_paths,
        semaphore=semaphore,
        stats=stats,
    )

    if injection_detected:
        result = dataclasses.replace(
            result, risk_flags=_merge_risk_flag(result.risk_flags, "text_instruction_present")
        )

    return result


async def run_pipeline(
    claims: list[dict],
    history_by_user: dict[str, dict],
    evidence_requirements: list[dict],
    images_base_dir: Path,
) -> tuple[list[ClaimOutput], RunStats]:
    """Process every claim concurrently, bounded by CONCURRENCY_LIMIT.

    Returns (results, stats) where stats is a RunStats accumulated across every
    call_vlm invocation made while processing this batch, for the operational-
    analysis report.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    stats = RunStats()
    start = time.monotonic()
    tasks = [
        process_claim(
            claim, history_by_user, evidence_requirements, images_base_dir, semaphore, stats
        )
        for claim in claims
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing claims")
    stats.runtime_seconds = time.monotonic() - start
    return results, stats
