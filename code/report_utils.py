"""Shared operational-analysis section formatting for evaluation_report.md.

Used by both evaluation/main.py (SAMPLE OPERATIONAL ANALYSIS, on sample_claims.csv)
and main.py (TEST OPERATIONAL ANALYSIS, on claims.csv) so the cost/TPM/RPM
formatting logic lives in one place.
"""

from __future__ import annotations

from vlm_client import RunStats

MODEL_NAME = "claude-sonnet-4-6"
INPUT_PRICE_PER_MTOK = 3.00
OUTPUT_PRICE_PER_MTOK = 15.00


def format_operational_section(title: str, stats: RunStats, row_count: int) -> list[str]:
    """Render one '## <title>' operational-analysis section as a list of lines."""
    cost = (stats.input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK + (
        stats.output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MTOK
    tokens_per_minute = (
        (stats.input_tokens + stats.output_tokens) / (stats.runtime_seconds / 60)
        if stats.runtime_seconds > 0
        else 0.0
    )
    rate_limit_note = (
        "rate limits were a factor (see 429/529 count above)"
        if stats.rate_limit_hits
        else "rate limits were not a factor"
    )

    return [
        f"## {title}",
        "",
        f"- Total rows processed: {row_count}",
        f"- Total API calls made: {stats.call_count}",
        f"- Total input tokens: {stats.input_tokens:,}",
        f"- Total output tokens: {stats.output_tokens:,}",
        f"- Images processed: {stats.images_processed}",
        f"- Cache hits vs API calls: {stats.cache_hits} hits / {stats.call_count} calls",
        f"- Actual runtime: {stats.runtime_seconds:.1f}s wall clock under "
        "`asyncio.Semaphore(5)` concurrency",
        f"- 429/529 responses hit: {stats.rate_limit_hits} "
        f"(retried {stats.retries} times with exponential backoff + jitter, "
        "bounded at 3 attempts per call; any other error fails fast)",
        f"- Approximate cost: ${cost:.4f} (assuming {MODEL_NAME} pricing: "
        f"${INPUT_PRICE_PER_MTOK:.2f}/MTok input, ${OUTPUT_PRICE_PER_MTOK:.2f}/MTok output)",
        f"- TPM/RPM notes: concurrency capped at `asyncio.Semaphore(5)` "
        "(`pipeline.CONCURRENCY_LIMIT`); observed "
        f"~{tokens_per_minute:,.0f} tokens/min during this run; {rate_limit_note}",
        "",
    ]
