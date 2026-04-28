# Quatrix — Q-Compass Architecture

> *"Where transformers retrieve by similarity, Quatrix navigates by value."*

**Quatrix** replaces standard multi-head attention with **Q-Compass** — a sequence-mixing primitive grounded in the reinforcement-learning $Q$-function rather than in geometric similarity. The same block runs across text, vision, audio, world-state transition, and cross-field tasks (cancer mutation signatures, drug-response, survival).

Built by **Syed Abdur Rehman Ali** ([@Abd0r](https://github.com/Abd0r)).

---

## Papers

Both PDFs are checked into [`Papers/`](./Papers).

1. **Q-Compass: Grounding Sequence Mixing in Reinforcement Learning Navigation** — [`Papers/qcompass.pdf`](./Papers/qcompass.pdf) · also on [Zenodo (March 2026)](https://zenodo.org/records/19104202). Defines the routing primitive (3-projection, no $W_V$).
2. **Quatrix: An Empirical Evaluation of Q-Compass and SAVO on Multimodal Sequence Modeling** — [`Papers/Quatrix.pdf`](./Papers/Quatrix.pdf). Multi-seed evaluation at 60M / 120M / 180M, KV-cache analysis, cross-field cancer demonstration (April 2026).

---

## Core Idea

### Q-Compass (3-projection, no $W_V$)

```
state  = x @ W_s          # "Where am I?"
action = x @ W_a          # "Where can I go?"
Q(s,a) = softmax(state @ action.T / sqrt(r))
output = W_o(Q(s,a) @ x)  # gather raw x — no W_V
```

Three projections ($W_s, W_a, W_o$). Value-based routing — *"in state s, how valuable is attending to position a?"*

### SAVO (4-projection variant: $Q$-value content)

SAVO reintroduces a $V$, but the $V$ projects the state⊙action product (a $Q$-value), not the raw input:

```
qval    = state ⊙ action                  # ∈ R^r, the Q-value vector
content = qval @ W_c                      # ∈ R^H, projected back up
output  = W_o(Q(s,a) @ content)
```

Four projections ($W_s, W_a, W_c, W_o$). Unlike standard attention's $W_V$ (linear map of raw $x$), SAVO's $W_c$ projects a $Q$-value — the $W_V$-free property is preserved at the *raw-input* level. The cost: $+rH$ parameters per block.

---

## Headline Empirical Results (paper §5.2 + §5.10)

| Comparison | Number | Recipe |
|---|---|---|
| SAVO vs rank-matched MHA, 60M (4-seed paired) | $+12.33 \pm 0.87$ ppl, $p = 7.6\!\times\!10^{-4}$ | 10k steps, identical hyperparameters |
| SAVO vs full-rank standard MHA, 60M (val) | $+5.79$ ppl above (worse) | full-rank MHA is parameter-undertrained at 10k steps |
| Rank-matched MHA vs full-rank MHA, 60M | $-6.54$ ppl below (better) | rank-matched 1×-attn-block converges; full-rank 8×-attn-block does not |
| KV-cache @ $r{=}H/8$ vs MHA | **0.125×** (matches MQA) | structural — content path is rank-$r$ by construction |
| KV-cache @ $r{=}H/16$ vs MHA | **0.0625×** (16× smaller) | $\le 1.6$ ppl penalty vs $r{=}H/8$ |
| Cross-field (cancer Phase 1–4) | parity within $\sim$5% of specialist baselines | same SAVO block, only I/O changes |

---

## Architecture

```
QuatrixLM (language model)
├── Token + Positional Embeddings
├── N × QuatrixBlock
│   ├── LayerNorm → QCompass (causal) → residual
│   └── LayerNorm → FFN (GELU) → residual
├── LayerNorm
└── Output Head (tied to embeddings)

QuatrixVision (image encoder)
├── Conv2d patch embedding (16×16 patches → 196 patches per 224×224 image)
├── Positional embeddings
├── M × QCompassBi blocks (bidirectional, no causal mask)
└── Linear projection → LM hidden dim

QuatrixAudio (audio encoder)
├── Mel-spectrogram patch embedding (16×16 freq×time patches)
├── 3 × QCompassBi blocks
└── Linear projection → LM hidden dim

QuatrixWorld (world-model plugin)
├── StateEncoder: QCompassBi aggregates a frame patch sequence → state vector
├── ActionHead: predicts action distribution from state
├── TransitionModel: 9–10 × QCompassBi blocks, predicts ŝ' = f(s, a)
└── RewardHead (optional): scalar value for RL fine-tuning

QuatrixWorldGenerative (frame-prediction world model)
└── Same Q-Compass block class, predicts the next FRAME (pixels), not just a latent state.

QuatrixCancerModel (cancer mutation-signature model)
└── SAVO stack ($H{=}384$, $r{=}48$, $h{=}6$) over SBS96 context vectors → softmax over signatures or cancer types.

QuatrixEditModel (gene-editing outcome predictor)
└── Architectural mirror of QuatrixWorldGenerative, applied to CRISPR edit outcomes.

TransformerLM (rank-matched MHA baseline, paper §5.2)
└── Standard 4-projection QKVO attention with all projections at rank r — apples-to-apples controlled ablation against SAVO.
```

---

## Repository Layout

```
quatrix/                         (repo root)
├── Papers/                      ← both papers
│   ├── Quatrix.pdf              ← April 2026 empirical paper (this repo)
│   └── qcompass.pdf             ← March 2026 original primitive paper
├── src/quatrix/                 ← Python package (importable as `import quatrix`)
│   ├── __init__.py              ← public API (QuatrixLM, QuatrixConfig, ...)
│   ├── config.py                ← QuatrixConfig dataclass
│   ├── model.py                 ← QCompass, QuatrixBlock, QuatrixLM (SAO + SAVO)
│   ├── vision.py                ← QCompassBi, VisionEncoder
│   ├── audio.py                 ← AudioEncoder, waveform_to_mel
│   ├── world.py                 ← WorldModel + StateEncoder + TransitionModel
│   ├── world_generative.py      ← QuatrixWorldGenerative (frame-prediction)
│   ├── cancer_model.py          ← QuatrixCancerModel (paper §7 Phase 1–4)
│   ├── edit_model.py            ← QuatrixEditModel (CRISPR-edit outcomes)
│   ├── transformer_lm.py        ← TransformerLM rank-matched MHA baseline (paper §5.2)
│   └── train.py                 ← python -m quatrix.train demo loop
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Modality Support

| Modality | Module | Block class | Attention mode |
|----------|--------|-------------|----------------|
| Text | `QuatrixLM` | `QCompass` | causal |
| Vision | `VisionEncoder` | `QCompassBi` | bidirectional |
| Audio | `AudioEncoder` | `QCompassBi` | bidirectional |
| World (latent) | `WorldModel` | `QCompassBi` | bidirectional |
| World (generative) | `QuatrixWorldGenerative` | `QCompassBi` | bidirectional |
| Cancer signatures | `QuatrixCancerModel` | `QCompassBi` (MH-QVC) | bidirectional |
| Gene-editing | `QuatrixEditModel` | `QCompassBi` | bidirectional |

---

## Quick Start

```bash
pip install quatrix
```

```python
from quatrix import QuatrixLM, QuatrixConfig
import torch

# Text only
cfg = QuatrixConfig(vocab_size=50257, hidden_size=512, num_layers=7,
                    max_seq_len=5120, q_rank=64)
model = QuatrixLM(cfg)
input_ids = torch.randint(0, 50257, (1, 10))
out = model(input_ids)
logits = out['logits']  # [B, L, vocab_size]

# Text + Vision
cfg = QuatrixConfig(vocab_size=50257, hidden_size=512, num_layers=7,
                    max_seq_len=5120, q_rank=64, use_vision=True)
model = QuatrixLM(cfg)
pixel_values = torch.randn(1, 3, 224, 224)
out = model(input_ids, pixel_values=pixel_values)

# Text + Vision + Audio
cfg = QuatrixConfig(vocab_size=50257, hidden_size=512, num_layers=7,
                    max_seq_len=5120, q_rank=64, use_vision=True, use_audio=True)
model = QuatrixLM(cfg)
mel = torch.randn(1, 1, 80, 3000)
out = model(input_ids, pixel_values=pixel_values, mel=mel)

# World Model
from quatrix import WorldModel
world = WorldModel(lm_hidden=512, action_dim=256)
hidden_states = model.get_hidden_states(input_ids)
state, action_logits, next_state, reward = world(hidden_states)
```

### Training

```bash
# Quick demo — TinyShakespeare, CPU/GPU
python -m quatrix.train

# Custom config
python -m quatrix.train --steps 2000 --hidden 512 --layers 7
python -m quatrix.train --data myfile.txt
```

---

## Roadmap

| Project | Description | Status |
|---|---|---|
| Q-Compass v1 | Routing primitive, 3-projection | Published (Zenodo) |
| Quatrix v1 (this repo) | SAVO 4-projection + multimodal evaluation + KV-cache analysis + cross-field demo | Empirical paper out |
| **NanoG1** | Cancer foundation model with mid-CoT hypothetical simulation, building on the Phase 1–4 setup in `cancer_model.py` | Future work |

---

## Citation

If you use Quatrix or Q-Compass in your work, please cite:

```bibtex
@misc{ali2026qcompass,
  author = {Syed Abdur Rehman Ali},
  title  = {Q-Compass: Grounding Sequence Mixing in Reinforcement Learning Navigation},
  year   = {2026},
  month  = {March},
  howpublished = {Zenodo},
  url = {https://zenodo.org/records/19104202}
}

@misc{ali2026quatrix,
  author = {Syed Abdur Rehman Ali},
  title  = {Quatrix: An Empirical Evaluation of Q-Compass and SAVO on Multimodal Sequence Modeling},
  year   = {2026},
  month  = {April},
  howpublished = {arXiv preprint},
  url = {https://github.com/Abd0r/quatrix}
}
```

---

## Author

**Syed Abdur Rehman Ali**

[![GitHub](https://img.shields.io/badge/GitHub-Abd0r-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abd0r)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Abd0r-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/Abd0r)
[![X](https://img.shields.io/badge/X%20%2F%20Twitter-SyedAbdurR2hman-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/SyedAbdurR2hman)

---

## License

OpenRAIL-M — open use with behavioral restrictions (no military use, no mass surveillance). See `LICENSE` for details.
