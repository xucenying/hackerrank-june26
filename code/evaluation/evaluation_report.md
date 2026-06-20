# Evaluation Report

## ACCURACY (dataset/sample_claims.csv only)

Run at: 2026-06-20T04:43:25+02:00
Evaluated on 20 rows from `dataset/sample_claims.csv`.

### Per-field accuracy

| field | accuracy | correct | wrong |
|---|---|---|---|
| `claim_status` | 85.0% | 17 | 3 |
| `evidence_standard_met` | 85.0% | 17 | 3 |
| `severity` | 50.0% | 10 | 10 |
| `valid_image` | 85.0% | 17 | 3 |
| `issue_type` | 45.0% | 9 | 11 |
| `object_part` | 85.0% | 17 | 3 |

### claim_status confusion matrix

| true \ predicted | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 1 | 0 |
| contradicted | 1 | 4 | 0 |
| not_enough_information | 1 | 0 | 1 |

### claim_status precision / recall / F1

| label | precision | recall | f1 |
|---|---|---|---|
| supported | 0.86 | 0.92 | 0.89 |
| contradicted | 0.80 | 0.80 | 0.80 |
| not_enough_information | 1.00 | 0.50 | 0.67 |

Macro F1: 0.79

### Risk flag precision: 73.9%

Fraction of predicted risk_flags tokens that appear in the expected risk_flags, over rows where the expected output actually has risk_flags.

### Rows where claim_status was wrong

| user_claim | predicted | expected |
|---|---|---|
| Customer: Hi, I found new damage on my car after it was parked outside overnight. \| Support: Sorry to hear that. Can you describe what changed? \| Customer: The back of the car has a dent now. It was not there before. \| Support: Did anything else break or is it mostly body damage? \| Customer: Mostly the rear bumper area. I attached the photo I took this morning. | contradicted | supported |
| Customer: The item I ordered was not inside the box. \| Support: Did the package look opened when you received it? \| Customer: I checked it after delivery and could not find the product inside. \| Support: What are you asking us to verify? \| Customer: Please verify that the contents are missing from the package. | supported | not_enough_information |
| Customer: My delivery box arrived opened. \| Support: Was the package crushed or was the seal affected? \| Customer: The seal area looked torn when I received it. \| Support: Are you asking us to review the package condition or the item inside? \| Customer: The package condition. I want the torn-open package reviewed. | supported | contradicted |

## SAMPLE OPERATIONAL ANALYSIS (dataset/sample_claims.csv)

- Total rows processed: 20
- Total API calls made: 0
- Total input tokens: 0
- Total output tokens: 0
- Images processed: 29
- Cache hits vs API calls: 20 hits / 0 calls
- Actual runtime: 0.7s wall clock under `asyncio.Semaphore(5)` concurrency
- 429/529 responses hit: 0 (retried 0 times with exponential backoff + jitter, bounded at 3 attempts per call; any other error fails fast)
- Approximate cost: $0.0000 (assuming claude-sonnet-4-6 pricing: $3.00/MTok input, $15.00/MTok output)
- TPM/RPM notes: concurrency capped at `asyncio.Semaphore(5)` (`pipeline.CONCURRENCY_LIMIT`); observed ~0 tokens/min during this run; rate limits were not a factor

## TEST OPERATIONAL ANALYSIS (dataset/claims.csv)

- Total rows processed: 44
- Total API calls made: 24
- Total input tokens: 80,649
- Total output tokens: 7,062
- Images processed: 82
- Cache hits vs API calls: 20 hits / 24 calls
- Actual runtime: 59.1s wall clock under `asyncio.Semaphore(5)` concurrency
- 429/529 responses hit: 0 (retried 0 times with exponential backoff + jitter, bounded at 3 attempts per call; any other error fails fast)
- Approximate cost: $0.3479 (assuming claude-sonnet-4-6 pricing: $3.00/MTok input, $15.00/MTok output)
- TPM/RPM notes: concurrency capped at `asyncio.Semaphore(5)` (`pipeline.CONCURRENCY_LIMIT`); observed ~89,007 tokens/min during this run; rate limits were not a factor

## TEST OPERATIONAL ANALYSIS (dataset/claims.csv)

- Total rows processed: 44
- Total API calls made: 0
- Total input tokens: 0
- Total output tokens: 0
- Images processed: 82
- Cache hits vs API calls: 44 hits / 0 calls
- Actual runtime: 1.7s wall clock under `asyncio.Semaphore(5)` concurrency
- 429/529 responses hit: 0 (retried 0 times with exponential backoff + jitter, bounded at 3 attempts per call; any other error fails fast)
- Approximate cost: $0.0000 (assuming claude-sonnet-4-6 pricing: $3.00/MTok input, $15.00/MTok output)
- TPM/RPM notes: concurrency capped at `asyncio.Semaphore(5)` (`pipeline.CONCURRENCY_LIMIT`); observed ~0 tokens/min during this run; rate limits were not a factor
