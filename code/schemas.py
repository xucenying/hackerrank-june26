"""Output field definitions and allowed values for Multi-Modal Evidence Review.

Mirrors problem_statement.md "Required output" / "Output meaning" / "Allowed values"
sections exactly. If those sections change, update this file to match.
"""

from __future__ import annotations

from dataclasses import dataclass

# Exact output column order required by problem_statement.md.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# --- Input-side values (echoed from claims.csv, not model-produced) ---

CLAIM_OBJECTS = {"car", "laptop", "package"}

# --- Allowed values for model/post-hoc-produced fields ---

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

OBJECT_PARTS_BY_CLAIM_OBJECT = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

# Union of all object parts, for contexts where claim_object is unknown/invalid.
ALL_OBJECT_PARTS = set().union(*OBJECT_PARTS_BY_CLAIM_OBJECT.values())

RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

SEVERITIES = {"none", "low", "medium", "high", "unknown"}

# evidence_standard_met / valid_image are written as the literal strings "true"/"false"
# (matching sample_claims.csv), not Python bools, so pandas doesn't capitalize them.
BOOL_STRINGS = {"true", "false"}

# Multi-value fields use ";" as the separator (risk_flags, supporting_image_ids),
# matching image_paths and sample_claims.csv.
LIST_SEPARATOR = ";"

# Sentinel used in supporting_image_ids when no image supports the decision.
NO_SUPPORTING_IMAGES = "none"

# Fallback row used when all submitted images are invalid/unreadable (Stage 1),
# or when Stage 3 enum coercion hard-fails after retries.
DEFAULT_NOT_ENOUGH_INFO = {
    "evidence_standard_met": "false",
    "evidence_standard_met_reason": "No usable image evidence was available to evaluate this claim.",
    "risk_flags": "none",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "No valid images were available, so the claim could not be evaluated.",
    "supporting_image_ids": "none",
    "valid_image": "false",
    "severity": "unknown",
}


@dataclass
class ClaimOutput:
    """One row of output.csv. Field order matches OUTPUT_COLUMNS."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str

    def as_row(self) -> dict:
        return {col: getattr(self, col) for col in OUTPUT_COLUMNS}
