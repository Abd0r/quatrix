# Full repaired study: all12 verified endpoints

Primary endpoint: validation token NLL at exactly20M valid targets. Mean +/- sample SD over four seeds.

| Model | NLL | Perplexity | Parameters |
|---|---:|---:|---:|
| Rank-matched MHA | 5.088288 +/- 0.007253 | 162.115 +/- 1.177 | 32,264,832 |
| SAVO | 5.134715 +/- 0.011558 | 169.824 +/- 1.964 | 33,555,072 |
| Sum-content control | 5.134017 +/- 0.007314 | 169.701 +/- 1.241 | 33,555,072 |

Per-seed values (NLL / ppl):

- Rank-matched MHA: 5.0905/162.5 | 5.0976/163.6 | 5.0827/161.2 | 5.0824/161.2
- SAVO: 5.1323/169.4 | 5.1366/170.1 | 5.1210/167.5 | 5.1489/172.3
- Sum-content control: 5.1249/168.2 | 5.1346/169.8 | 5.1428/171.2 | 5.1338/169.7

Paired seed-level NLL differences:

- savo_minus_rankmatched: mean +0.046427, sample SD 0.013506, approx t95 CI [+0.024936, +0.067917], all four positive: True
- savo-sum_minus_rankmatched: mean +0.045729, sample SD 0.012161, approx t95 CI [+0.026377, +0.065080], all four positive: True
- savo_minus_savo-sum: mean +0.000698, sample SD 0.015962, approx t95 CI [-0.024700, +0.026097], all four positive: False

CI caveat: approximate t interval over n=4 seed differences; assumes approximately normal differences (untestable at n=4). Seeds are the replication unit, not documents. No significance claim; no p-values reported; no metric or seed selection.

Design caveats: Rank-matched MHA has fewer parameters than SAVO/sum-content (not parameter-matched). SAVO and sum-content control are parameter-count matched but activation distributions are not fully matched. This fixed recipe is not optimally tuned per architecture. Separate from the historical10k-step study; not a reproduction of it. 16 exact train/test overlaps removed from training; near-duplicate decontamination incomplete.
