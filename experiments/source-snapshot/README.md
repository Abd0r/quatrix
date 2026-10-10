# Source snapshot for the recorded experiment logs

The run manifests in `../repaired-main/` record the SHA256 of the source version that produced
them. This directory holds that exact version so the hashes stay verifiable.

| File | Role |
|---|---|
| `src/quatrix/model.py` | model, Q-Compass (SAO), SAVO, world branch |
| `src/quatrix/transformer_lm.py` | rank-matched MHA comparator |
| `src/quatrix/config.py` | model configuration |
| `driver/train_repaired_main.py` | the fixed-budget training driver |
| `driver/evaluation.py` | the length-masked, token-weighted scorer |
| `local-modification.patch` | the working-tree edit applied on top of the base commit |

`hashes.json` lists the five SHA256 values as recorded in the manifests.

## Verify

    cd experiments/source-snapshot
    python3 - <<'EOF'
    import hashlib, json, pathlib
    h = json.load(open("hashes.json"))["sha256"]
    for rel, want in h.items():
        got = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
        print("MATCH " if got == want else "DIFFER", rel)
    EOF

## Why this is not the same as the current source tree

Current `master` carries input-validation guards added after the runs finished
(`raise ValueError` when the sequence length exceeds `max_seq_len`). Those commits touch
`model.py`, `transformer_lm.py` and six other files, but they add guards only: no attention math,
loss, masking or initialization code changed, and the experiments ran at exactly the 1024 maximum
context, so no measured value is affected. `config.py` is unchanged and still matches master.

This snapshot is the version the numbers came from. The project source tree is the version under
development.
