# Evaluation Report

Evaluated on 20 rows from `dataset/sample_claims.csv`.

## Per-field accuracy

- `evidence_standard_met`: 65.0%
- `evidence_standard_met_reason`: 0.0%
- `risk_flags`: 15.0%
- `issue_type`: 45.0%
- `object_part`: 90.0%
- `claim_status`: 70.0%
- `claim_status_justification`: 0.0%
- `supporting_image_ids`: 75.0%
- `valid_image`: 70.0%
- `severity`: 40.0%

## claim_status precision / recall / F1

| label | precision | recall | f1 |
|---|---|---|---|
| contradicted | 1.00 | 0.40 | 0.57 |
| not_enough_information | 0.29 | 1.00 | 0.44 |
| supported | 0.91 | 0.77 | 0.83 |

Macro F1: 0.62

## Operational analysis

- Model calls (sample run, 20 claims): 20 (20 Stage 2 calls + 0 Stage 3a enum-fix retries)
- Images processed (sample run): 29
- Token usage (sample run): 43,087 input / 5,115 output (avg 2,154 in / 256 out per call)
- Measured cost (sample run): $0.2060 (assuming Claude Sonnet 4.6 pricing: $3.00/MTok input, $15.00/MTok output)
- Runtime (sample run): 31.0s wall clock under asyncio.Semaphore(5) concurrency

### Projected for the full test set (44 claims in `dataset/claims.csv`)

- Projected model calls: ~44 (linear scale-up of the sample's 1.00 calls/claim, including expected enum-fix retries)
- Projected cost: ~$0.45
- Projected runtime: ~68s wall clock at the same concurrency

### Rate limits, batching, and retry strategy

- Concurrency is capped at `asyncio.Semaphore(5)` (`pipeline.CONCURRENCY_LIMIT`) to stay well under per-minute request and token limits for a single API key; raise this only after checking the account's actual RPM/TPM tier.
- `vlm_client.call_vlm` retries only on HTTP 429 (rate limited) and 529 (overloaded) with exponential backoff plus jitter, max 3 attempts; other errors fail fast rather than masking a real problem behind retries.
- No request-level caching or batching is implemented in this first benchmark pass -- each claim is one independent Stage 2 call. Since the system prompt is identical across every claim, adding `cache_control` to it would cut input-token cost on every call after the first; the Batches API (50% cost) would also fit well here since this is an offline, non-interactive job with no per-claim latency requirement.
- Stage 3a's enum-fix retry is bounded to exactly one extra call per invalid field, so a single malformed claim cannot cascade into unbounded extra spend.
