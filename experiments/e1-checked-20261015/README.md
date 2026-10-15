# Checked E1 and SAVO continuation evidence

A dated snapshot, not an automatically synchronized mirror. See `VERIFICATION.json` for the exact UTC capture time and `LOG-MANIFEST.json` for every released file/hash. Earlier `experiments/repaired-main` records are preserved, not replaced.

## Contents and status

| Records | Completed at snapshot | Interpretation |
|---|---:|---|
| `main-v3-text-only-seed*`, `main-v3-joint-4-seed*` | 8 | Original E1 uses **SAO**, not SAVO. Four paired20M WikiText-budget runs. Joint models have extra modality targets and more optimizer updates. |
| `frozen-joint4-seed*` | 4 | Preserved outputs under executed-driver source provenance review. A later edited helper regression did **not** establish that these historical runs were invalid. No causal freeze claim here. |
| `vision-only-seed*`, `audio-only-seed*` | 4 | Single-modality references; recorded text validation is included. |
| `world-only-seed*` | 2 | Both reached300,000steps but retained **no checkpoint and no validation endpoint**. Training logs only, not scored/recoverable world models. |
| `stepmatched2-textonly-seed*` | 4 | SAO sensitivities at17,701/17,905/17,942/17,862updates. They consume about57-58Mtext targets, so update count and text exposure both change. |
| `savo-core-clean-joint4-seed17` | 1 | Exactly20M WikiText targets; SAVO in LM blocks only. New modality populations enforce exclusions. Branch mixers remain unchanged. |
| `savo-next-v1/e2-clean-seed17.json` | 1 seed | 400vision+400audio examples, real/shuffled/zero/none; zero is an encoded-feature prefix. Shuffled donors are a one-to-one identity-distinct permutation. |
| `savo-next-v1/world-clean-seed17.json` | 1 seed | All6,431test transitions from6episodes excluded from the new training population. Encoded-state MSE, not pixel reconstruction or action-selection quality. |
| `savo-next-v1/cache-reference-seed17.json` | 1 checkpoint | Actual FP32 CUDA same-checkpoint comparison: cached teacher-forced16-token prediction versus full-prefix recomputation, at64/256/456prefix tokens. Not transformer-versus-SAVO speed. |
| `savo-next-v1/flash-savo-padded-seed17.json` | Blocked | Kernel probe exceeded shared memory (135,168bytes required,101,376available). **No Flash timing or speedup result.** |
| Other `e2-*.json` | Preserved diagnostics | Old caption/audio/world E2 subsets belonged to original training populations: **in-pool**, not held-out generalization. Text validation remains separately held out. |

The active SAVO seed29 run was excluded; seeds43/71 had not started at snapshot. This is not a completed four-seed SAVO multimodal comparison. Do not pool a single new seed with the historical SAO cohort as an architecture-only test: the data populations and recipe scope differ.

## What was checked

- Every completed run's `training.jsonl` rows match its `result.json` step log.
- Completed step/token budgets, source/driver hashes as recorded, and final endpoint consistency were checked.
- For21runs with retained endpoints, the **physical checkpoint SHA256** was recomputed and matched the recorded digest. Weight files are not published.
- Text validation uses1,038rows/194,461valid targets; per-document losses recompute the aggregate.
- Clean prefix and world results were independently checked for sample/condition counts, finite losses, permutation/identity exclusions, unique transitions and partition binding.
- Cache logits matched within3.9e-5maximum absolute error and stored tensors matched the declared2raccounting. Timing samples are reported, not seed uncertainty. No1M-context quality claim.

Historical frozen-driver hashes are recorded, but exact matching executed driver source remains unrecovered. Recorded hashes do not by themselves authenticate an unavailable implementation.

## Repair history and failed probes

The first cache launch requested512prefix tokens plus15teacher-forced input tokens, but the authenticated validation corpus has maximum row length471. The continuation uses64/256/456instead; no model capacity/data concatenation change or result-based tuning.

The first clean-prefix evaluation failed because the donor routine sorted row indices then assigned rotated donors using unsorted recipient positions. The versioned repair maps `zip(ordered, rotated)`. Regression tests reproduce the real failing400-row sample and cover all8task/seed populations. Earlier failure console/state records remain in `savo-next-v1`; the successful seed17 evaluation uses the corrected version3 evaluator. Completed training was not repeated.

## Paths, hashes and compression

Machine-home paths and private network addresses are redacted. Numeric measurements, seeds, source/checkpoint hash strings and dataset group identifiers are unchanged. Accordingly, **original artifact hash**, **released decoded hash**, and **stored file hash** are separate fields in the index. Hash references inside redacted manifests still refer to original artifacts, not silently to rewritten public bytes.

Files larger than4MiB use deterministic lossless gzip (`mtime=0`), keeping all recorded steps without exceeding GitHub file-size limits. Read with `gzip -dc FILE.json.gz` or Python's `gzip.open`. No weights, raw media, credentials, transport helpers or private notes are included.

This logs/results bundle is **not full executable training reproduction**: it records outputs and hashes; the prior source snapshot remains separate. No general environment-adaptation, planning, transfer, clinical utility or family-wide architectural ceiling is established by these records.
