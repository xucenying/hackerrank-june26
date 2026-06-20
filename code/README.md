# Multi-Modal Evidence Review -- Solution

A benchmark implementation for the HackerRank Orchestrate evidence-review challenge.
For each claim, one Claude Sonnet 4.6 vision call inspects the submitted images
against the claim conversation and produces the 10 required output fields, which
are then validated and normalized in a deterministic post-hoc pass.

## Install

```bash
pip install anthropic pandas python-dotenv tqdm pillow
```

Create a `.env` file in the repo root (never commit it):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

From the repo root:

```bash
python code/main.py --input dataset/claims.csv --output output.csv
```

Optional flags:

- `--dataset-dir` -- defaults to `dataset/` (must contain `user_history.csv`,
  `evidence_requirements.csv`, and `images/`).
- `--output` -- defaults to `output.csv` in the repo root.

## Evaluate

Runs the same pipeline against the labeled `dataset/sample_claims.csv`, scores the
predictions, and writes a report:

```bash
python code/evaluation/main.py
```

Produces:

- `code/evaluation/sample_predictions.csv` -- predictions for the sample set
- `code/evaluation/evaluation_report.md` -- per-field accuracy, `claim_status`
  precision/recall/F1, and an operational analysis (model calls, token usage,
  cost estimate, runtime, rate-limit/retry strategy)

## Architecture

Three stages per claim, run concurrently under `asyncio.Semaphore(5)`:

1. **Load inputs** (`pipeline.py`, `image_utils.py`) -- join `user_history.csv`
   by `user_id`, filter `evidence_requirements.csv` by `claim_object`, and load +
   base64-encode every image in `image_paths`. If every image fails to load, the
   claim short-circuits to a `not_enough_information` / `unknown` fallback row
   with no model call.
2. **VLM analysis** (`vlm_client.py`, `prompts.py`) -- one Claude Sonnet 4.6
   vision call per claim, with the claim conversation, the relevant evidence
   requirements, and the user's history context. The model returns a single
   JSON object with the 10 output fields. Retries on HTTP 429/529 only, with
   exponential backoff (max 3 attempts).
3. **Post-hoc validation** (`post_hoc.py`) -- no model call except a bounded
   one-shot retry per invalid enum field:
   - normalizes every field and checks it against the allowed values in
     `schemas.py`; on a still-invalid field, asks the model to fix that one
     field once, then falls back to a safe default if it's still invalid
   - enforces logical consistency: `valid_image=false` forces
     `evidence_standard_met=false`; `evidence_standard_met=false` forces
     `claim_status` away from `supported`
   - restricts `supporting_image_ids` to image IDs that were actually loaded
   - merges in `user_history_risk` when `past_claim_count > 5`,
     `rejected_claim > 0`, or `history_flags` is set

## File structure

```
code/
  main.py              # CLI entry point
  pipeline.py           # Stage 1-3 orchestration, concurrency
  vlm_client.py         # Anthropic SDK wrapper, retries, usage tracking
  image_utils.py        # image loading / base64 (stdlib only)
  schemas.py            # output columns + allowed values, from problem_statement.md
  prompts.py            # all prompt strings
  post_hoc.py           # Stage 3 validation logic
  evaluation/
    main.py             # runs the pipeline against sample_claims.csv, scores it
    metrics.py          # per-field accuracy, claim_status F1
    evaluation_report.md  # written by evaluation/main.py
```

## Design notes / known limitations

- This is a first benchmark pass, optimized for clarity over cost. There is no
  prompt caching or batching yet -- see the "Rate limits, batching, and retry
  strategy" section of `evaluation_report.md` for what the natural next
  optimizations would be (caching the system prompt, the Batches API for the
  offline `claims.csv` run).
- `supporting_image_ids` reflects whatever image(s) the model grounds its
  decision in, regardless of `claim_status` -- including for `contradicted`
  claims, where the contradicting image is the relevant evidence. There is no
  rule forcing `supporting_image_ids=none` on a contradicted claim.
- A claim's image-quality and authenticity risk flags (e.g. `non_original_image`,
  `text_instruction_present`) are produced by the model's own visual judgment;
  the post-hoc pass only adds `user_history_risk` from the numeric/flag fields
  in `user_history.csv`.
