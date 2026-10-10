# Verified legacy text reevaluation

44 evaluations across 22 checkpoints. Mean +/- sample SD, not SE.

| 60m family model | Runs | Validation ppl | Test ppl |
|---|---:|---:|---:|
| rankmatched | 4 | 251.422 +/- 0.393 | 250.297 +/- 0.652 |
| savo | 4 | 263.749 +/- 0.683 | 262.458 +/- 0.550 |
| fullrank | 2 | 257.960 +/- 2.124 | 257.110 +/- 1.731 |

SAVO minus rank-matched validation mean: +12.327 ppl (worse).

## Interpretation and limits
- These independently rerun scores largely reproduce historical outputs; a flaw in another evaluator did not invalidate the verified table.
- Family names are not actual text-only parameter counts. Exact counts are in all-evaluations.csv.
- Same run-index does not establish paired data ordering. The original seed-0 path was unseeded. No paired significance claim is retained.
- Full-rank MHA has two recovered runs, not four; no convergence or undertraining mechanism is established by its ranking.
- Training budgets, micro-batching and parameter counts are not fully controlled across all baselines.
- The corpus is a filtered derived subset with custom preprocessing, not canonical WikiText-103 scoring.

## Test contamination sensitivity
- 16 exact normalized train/test overlap rows identified. Both original and consistently excluded scores are recorded. No tuning uses test outcomes.
- Near-duplicate and semantic contamination checks remain incomplete.
