"""Stage 3: enum coercion, logical consistency, and risk-flag merge -- no model
call except the bounded one-shot enum-fix retry in _request_fix().
"""

from __future__ import annotations

import asyncio
import json

from image_utils import LoadedImage
from prompts import ENUM_FIX_PROMPT_TEMPLATE, SYSTEM_PROMPT
from schemas import (
    ALL_OBJECT_PARTS,
    BOOL_STRINGS,
    CLAIM_STATUSES,
    DEFAULT_NOT_ENOUGH_INFO,
    ISSUE_TYPES,
    LIST_SEPARATOR,
    NO_SUPPORTING_IMAGES,
    OBJECT_PARTS_BY_CLAIM_OBJECT,
    RISK_FLAGS,
    SEVERITIES,
    ClaimOutput,
)
from vlm_client import call_vlm


def _split_normalize(raw: object) -> list[str]:
    return [t.strip().lower() for t in str(raw).split(LIST_SEPARATOR) if t.strip()]


def _coerce_text(raw: object, default: str) -> str:
    text = str(raw).strip() if raw is not None else ""
    return text if text else default


async def _request_fix(
    field_name: str,
    invalid_value: object,
    allowed: set[str],
    model_output: dict,
    semaphore: asyncio.Semaphore,
) -> object | None:
    """One-shot retry: ask the model to correct a single invalid field."""
    fix_prompt = ENUM_FIX_PROMPT_TEMPLATE.format(
        field_name=field_name,
        invalid_value=invalid_value,
        allowed_values=", ".join(sorted(allowed)),
        previous_response=json.dumps(model_output),
    )
    async with semaphore:
        try:
            fixed_response = await call_vlm(SYSTEM_PROMPT, fix_prompt, images=[])
        except Exception:
            return None
    return fixed_response.get(field_name)


async def _coerce_scalar(
    field_name: str,
    raw: object,
    allowed: set[str],
    model_output: dict,
    semaphore: asyncio.Semaphore,
) -> str:
    normalized = str(raw).strip().lower() if raw is not None else ""
    if normalized in allowed:
        return normalized

    fixed = await _request_fix(field_name, raw, allowed, model_output, semaphore)
    if fixed is not None:
        fixed_normalized = str(fixed).strip().lower()
        if fixed_normalized in allowed:
            return fixed_normalized

    return DEFAULT_NOT_ENOUGH_INFO[field_name]


async def _coerce_list(
    field_name: str,
    raw: object,
    allowed: set[str],
    model_output: dict,
    semaphore: asyncio.Semaphore,
) -> str:
    tokens = _split_normalize(raw)
    if tokens and all(token in allowed for token in tokens):
        return LIST_SEPARATOR.join(dict.fromkeys(tokens))

    fixed = await _request_fix(field_name, raw, allowed, model_output, semaphore)
    if fixed is not None:
        fixed_tokens = _split_normalize(fixed)
        if fixed_tokens and all(token in allowed for token in fixed_tokens):
            return LIST_SEPARATOR.join(dict.fromkeys(fixed_tokens))

    return DEFAULT_NOT_ENOUGH_INFO[field_name]


def _restrict_to_valid_ids(value: str, valid_ids: set[str]) -> str:
    """supporting_image_ids must only reference images that were actually loaded."""
    if value == NO_SUPPORTING_IMAGES:
        return value
    tokens = [t for t in value.split(LIST_SEPARATOR) if t in valid_ids]
    return LIST_SEPARATOR.join(tokens) if tokens else NO_SUPPORTING_IMAGES


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _history_indicates_risk(history: dict) -> bool:
    past_claim_count = _safe_int(history.get("past_claim_count"))
    rejected_claim = _safe_int(history.get("rejected_claim"))
    history_flags = str(history.get("history_flags", "none")).strip().lower()
    return past_claim_count > 5 or rejected_claim > 0 or history_flags not in ("", "none")


def _merge_history_risk(risk_flags: str, history: dict) -> str:
    tokens = [] if risk_flags == "none" else risk_flags.split(LIST_SEPARATOR)
    if _history_indicates_risk(history) and "user_history_risk" not in tokens:
        tokens.append("user_history_risk")
    tokens = list(dict.fromkeys(tokens))
    return LIST_SEPARATOR.join(tokens) if tokens else "none"


async def apply_post_hoc(
    claim: dict,
    history: dict,
    model_output: dict,
    valid_images: list[LoadedImage],
    invalid_paths: list[str],
    semaphore: asyncio.Semaphore,
) -> ClaimOutput:
    """Stage 3: coerce model_output into allowed values, enforce logical
    consistency between fields, and merge in history-derived risk flags."""
    valid_image_ids = {img.image_id for img in valid_images}
    claim_object = claim["claim_object"]
    object_part_allowed = OBJECT_PARTS_BY_CLAIM_OBJECT.get(claim_object, ALL_OBJECT_PARTS)
    supporting_ids_allowed = valid_image_ids | {NO_SUPPORTING_IMAGES}

    evidence_standard_met = await _coerce_scalar(
        "evidence_standard_met", model_output.get("evidence_standard_met"), BOOL_STRINGS, model_output, semaphore
    )
    valid_image = await _coerce_scalar(
        "valid_image", model_output.get("valid_image"), BOOL_STRINGS, model_output, semaphore
    )
    claim_status = await _coerce_scalar(
        "claim_status", model_output.get("claim_status"), CLAIM_STATUSES, model_output, semaphore
    )
    issue_type = await _coerce_scalar(
        "issue_type", model_output.get("issue_type"), ISSUE_TYPES, model_output, semaphore
    )
    object_part = await _coerce_scalar(
        "object_part", model_output.get("object_part"), object_part_allowed, model_output, semaphore
    )
    severity = await _coerce_scalar(
        "severity", model_output.get("severity"), SEVERITIES, model_output, semaphore
    )
    risk_flags = await _coerce_list(
        "risk_flags", model_output.get("risk_flags"), RISK_FLAGS, model_output, semaphore
    )
    supporting_image_ids = await _coerce_list(
        "supporting_image_ids",
        model_output.get("supporting_image_ids"),
        supporting_ids_allowed,
        model_output,
        semaphore,
    )

    evidence_standard_met_reason = _coerce_text(
        model_output.get("evidence_standard_met_reason"),
        DEFAULT_NOT_ENOUGH_INFO["evidence_standard_met_reason"],
    )
    claim_status_justification = _coerce_text(
        model_output.get("claim_status_justification"),
        DEFAULT_NOT_ENOUGH_INFO["claim_status_justification"],
    )

    # Logical consistency.
    if valid_image == "false":
        evidence_standard_met = "false"
    if evidence_standard_met == "false" and claim_status == "supported":
        claim_status = "not_enough_information"

    supporting_image_ids = _restrict_to_valid_ids(supporting_image_ids, valid_image_ids)
    risk_flags = _merge_history_risk(risk_flags, history)

    return ClaimOutput(
        user_id=claim["user_id"],
        image_paths=claim["image_paths"],
        user_claim=claim["user_claim"],
        claim_object=claim_object,
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_standard_met_reason,
        risk_flags=risk_flags,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=claim_status_justification,
        supporting_image_ids=supporting_image_ids,
        valid_image=valid_image,
        severity=severity,
    )
