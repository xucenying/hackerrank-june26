"""Anthropic SDK wrapper for the VLM pipeline: vision calls with retry/backoff.

pipeline.py should call into this module only -- raw `anthropic` SDK usage should
not appear elsewhere. Concurrency control (the asyncio.Semaphore(5)) lives in
pipeline.py, not here; this module makes one call per invocation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random

from anthropic import APIStatusError, AsyncAnthropic

from image_utils import LoadedImage

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 529}
BASE_BACKOFF_SECONDS = 1.0
MAX_OUTPUT_TOKENS = 1024

_client: AsyncAnthropic | None = None

# Running totals across calls, for the operational-analysis report
# (evaluation/evaluation_report.md). Not thread-safe beyond asyncio's
# single-threaded event loop -- fine here since all callers await call_vlm.
_usage_stats = {"call_count": 0, "input_tokens": 0, "output_tokens": 0}


def reset_usage_stats() -> None:
    _usage_stats.update(call_count=0, input_tokens=0, output_tokens=0)


def get_usage_stats() -> dict:
    return dict(_usage_stats)


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
    system_prompt: str,
    user_text: str,
    images: list[LoadedImage],
) -> dict:
    """Send one call: system prompt + optional images + user_text. Returns the
    parsed JSON object from the model's reply.

    Used both for the Stage 2 claim-analysis call (with images) and the Stage 3a
    enum-fix retry (text only, images=[]).

    Retries on HTTP 429 (rate limited) and 529 (overloaded) with exponential
    backoff plus jitter, up to MAX_RETRIES attempts. Any other API error, or a
    non-JSON reply, is raised immediately to the caller.
    """
    client = get_client()
    content: list[dict] = _image_blocks(images) + [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": content}]

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=messages,
            )
            _usage_stats["call_count"] += 1
            _usage_stats["input_tokens"] += response.usage.input_tokens
            _usage_stats["output_tokens"] += response.usage.output_tokens
            text = "".join(block.text for block in response.content if block.type == "text")
            return _extract_json(text)
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
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
