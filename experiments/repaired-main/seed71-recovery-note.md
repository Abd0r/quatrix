# Seed71 sum-content recovery

The first `savo-sum-seed71-20m` run was interrupted during its attended SSH invocation. Read-only inspection afterward found no final result.json, a last recorded step5300 /17070142 supervised targets, and no active train_repaired_main.py process. Its artifacts are preserved, not overwritten or discarded based on performance.

Recovery: rerun the same frozen train_repaired_main.py from scratch with variant savo-sum, seed71, target20000000 and all other original defaults, writing to the NEW remote directory `results/savo-sum-seed71-20m-recovery1`. This is a transparently named replacement for an interrupted run, not an additional selected seed or a hyperparameter change. No checkpoint resumption is claimed. Use the recovery's completed final endpoint in the declared seed71 comparison only after hashes, budget, sampler and manifest are verified. Compare the repeated training prefix with the preserved partial run and report any discrepancy.

Original remote root: `/home/abdur/quatrix-tmlr/run-20261009-repaired-main`. No original code/data or partial outputs are changed. Compilation remains blocked and will not be retried.

Recovery invocation subsequently lost its SSH connection (exit255) after transmitted step5800 /18682323 targets. A follow-up connection timed out. Whether the remote process completed is unknown until SSH is restored and result/process state is checked. Do not launch a third run based on this transport failure. The command output is preserved in tests/remote-main-savo-sum-seed71-recovery1.log. The replayed step5300 output matches the original visible NLL/gradient norm exactly, but no full-prefix or final-checkpoint equality is asserted without retrieval.
