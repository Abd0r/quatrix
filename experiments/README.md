# Quatrix experiment logs

Recorded outputs of the Quatrix experiments. **Logs and results only.** The model implementation
is in [`src/quatrix`](../src/quatrix); exact experimental driver/source versions are identified by
the recorded hashes and separate snapshots. Do not assume current source reproduces every historical run.

**Latest checked snapshot:** [`e1-checked-20261015/`](e1-checked-20261015/README.md):23completed
E1/follow-up training records, independent endpoint/hash checks, clean seed17 evaluations,
cache benchmark and blocked Flash probe. Active/unstarted SAVO seeds are not completed evidence.

## What is here

| Directory | Contents |
|---|---|
| `e1-checked-20261015/` | Dated checked E1/SAVO continuation snapshot; status/caveats and per-file provenance in its README and LOG-MANIFEST. Large logs use lossless gzip. |
| `repaired-main/` | The predeclared fixed-budget study: 12 completed runs (4 seeds x rank-matched MHA, SAVO, sum-content control) plus the preserved recovery3 run and the 3333-target smoke test. Each run directory holds `result.json` (per-step log: step, targets, train NLL, gradient norm), four validation snapshots at ~5M/10M/15M/20M targets, and `manifest.json` (recipe, data and source hashes, environment). |
| `full-study-v1/` | Deterministic aggregate of the 12 endpoints: `summary.json`, `individual-runs.csv`, `main-table.tex`, `summary.md`, `per-seed-nll.png`. |
| `verified-summary-v1/` | Re-scored legacy text checkpoints (length-masked, token-weighted scorer) with the generated core table. |
| `legacy-recheck/` | Per-checkpoint legacy validation/test re-evaluations. |
| `multimodal-text-path/` | Text-path evaluations of multimodal-trained checkpoints. |
| `legacy-inventory/` | Inventory of recovered legacy checkpoints. |
| `legacy-training-logs/` | The legacy training logs that were recovered. **Partial**: three files of about 50 lines each. |

Also present at the top level: `full-study-verification.json`, the per-seed verification receipts,
`seed71-recovery-note.md` and the seed71 status files.

## Honest scope of this record

- The repaired fixed-budget study is **fully recorded**: every run has a per-step log, four
  validation snapshots, a manifest with data/source hashes, and a sampler hash.
- The legacy study is **not fully recorded**. Only three training logs were recovered, and the
  original multimodal run whose endpoints were transcribed in the first paper is lost. The
  recovered 60m multimodal log belongs to a different run and must not be mixed with it.
- The seed71 sum-content endpoint took four attempts. The original and two reruns were interrupted;
  only `savo-sum-seed71-20m-recovery3` reached the declared 20,000,000-target budget. The
  interruption cause was never established, and the earlier attempts left no local artifacts.
- Checkpoint weights are not included, only their SHA256 hashes as recorded in the manifests.
- No test-split scoring and no result-based tuning were performed in the repaired study.

## Reading an original repaired text run

`manifest.json` holds the declared recipe: width 384, ten layers, rank 48, FFN ratio 4, context
1024, tied embeddings, dropout 0.1, effective batch 16, micro-batch 2, AdamW at 3e-4 with betas
(0.9, 0.95) and weight decay 0.01, constant LR, BF16 forward with FP32 parameters and
cross-entropy, exactly 20M supervised targets. `result.json` holds the per-step log, the four
checkpoint records, and the final validation result (1038 paragraphs, 194461 supervised targets).

## Licensing

Released under the repository LICENSE, matching the CC BY 4.0 terms of the Quatrix preprint.
