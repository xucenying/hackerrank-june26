"""Anthropic SDK wrapper for the VLM pipeline: vision calls with retry/backoff.

pipeline.py should call into this module only -- raw `anthropic` SDK usage should
not appear elsewhere. Concurrency control (the asyncio.Semaphore(5)) lives in
pipeline.py, not here; this module makes one call per invocation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
from pathlib import Path

from anthropic import APIStatusError, AsyncAnthropic

import prompts
from image_utils import LoadedImage

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 529}
BASE_BACKOFF_SECONDS = 1.0
MAX_OUTPUT_TOKENS = 1024

CACHE_DIR = Path(__file__).resolve().parent / "cache"

_client: AsyncAnthropic | None = None


@dataclasses.dataclass
class RunStats:
    """Per-run instrumentation, threaded through call_vlm by the caller (pipeline.py,
    post_hoc.py) and surfaced by run_pipeline for the operational-analysis report."""

    call_count: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images_processed: int = 0
    rate_limit_hits: int = 0
    retries: int = 0
    runtime_seconds: float = 0.0


def get_client() -> AsyncAnthropic:
    """Lazily construct a single shared AsyncAnthropic client from ANTHROPIC_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (expected via .env)")
        _client = AsyncAnthropic(api_key=api_key)
    return _client


def _image_blocks(images: list[LoadedImage]) -> list[dict]:
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.base64_data,
            },
        }
        for image in images
    ]


def _cache_key(system_prompt: str, user_text: str, images: list[LoadedImage]) -> str:
    """sha256 of concatenated image bytes (base64) + the full prompt text (system + user)."""
    hasher = hashlib.sha256()
    for image in images:
        hasher.update(image.base64_data.encode("ascii"))
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(user_text.encode("utf-8"))
    return hasher.hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cached(key: str) -> dict | None:
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data), encoding="utf-8")


def _extract_json(text: str) -> dict:
    """Parse the model's reply as a single JSON object, tolerating stray markdown
    fences if the model adds them despite SYSTEM_PROMPT instructing otherwise."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


async def call_vlm(
    claim_object: str,
    user_text: str,
    images: list[LoadedImage],
    stats: RunStats,
) -> dict:
    """Send one call: claim_object-scoped system prompt + optional images + user_text.
    Returns the parsed JSON object from the model's reply.

    Used both for the Stage 2 claim-analysis call (with images) and the Stage 3a
    enum-fix retry (text only, images=[]). The system prompt is built from
    prompts.get_prompt(claim_object), which scopes the allowed issue_type and
    object_part values to that claim_object.

    Retries on HTTP 429 (rate limited) and 529 (overloaded) with exponential
    backoff plus jitter, up to MAX_RETRIES attempts. Any other API error, or a
    non-JSON reply, is raised immediately to the caller.
    """
    system_prompt = prompts.get_prompt(claim_object)
    client = get_client()
    content: list[dict] = _image_blocks(images) + [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": content}]

    cache_key = _cache_key(system_prompt, user_text, images)
    cached = _load_cached(cache_key)
    if cached is not None:
        stats.cache_hits += 1
        return cached

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=messages,
            )
            stats.call_count += 1
            stats.input_tokens += response.usage.input_tokens
            stats.output_tokens += response.usage.output_tokens
            text = "".join(block.text for block in response.content if block.type == "text")
            parsed = _extract_json(text)
            _save_cache(cache_key, parsed)
            return parsed
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code in RETRYABLE_STATUS_CODES:
                stats.rate_limit_hits += 1
            if exc.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
            stats.retries += 1
            delay = BASE_BACKOFF_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "VLM call failed with status %s (attempt %d/%d), retrying in %.1fs",
                exc.status_code,
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
