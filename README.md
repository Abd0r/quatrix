# Quatrix — Q-Compass Architecture

> *"Where transformers retrieve by similarity, Quatrix navigates by value."*

**Quatrix** is a novel neural architecture that replaces standard multi-head attention with **Q-Compass** — a sequence mixing mechanism grounded in reinforcement learning theory rather than geometric similarity.

Built by **Syed Abdur Rehman Ali** ([@Abd0r](https://github.com/Abd0r)).

**Paper:** [Q-Compass: Grounding Sequence Mixing in Reinforcement Learning Navigation](https://zenodo.org/records/19104202) — Zenodo, March 2026.

---

## Core Idea: Q-Compass

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
```
Four projections (W_Q, W_K, W_V, W_O). Similarity-based routing — attends to what *looks similar*, retrieves a projected transform of it.

**Q-Compass computes:**
```
state  = x @ W_s          # "Where am I?"
action = x @ W_a          # "Where can I go?"
Q(s,a) = softmax(state @ action.T / sqrt(r))
output = W_o(Q(s,a) @ x)  # gather from x directly — no W_V
```
Three projections (W_s, W_a, W_o). Value-based routing — asks *"in state s, how valuable is attending to position a?"*

**The key removal:** No W_V. Content is gathered directly from `x`, unchanged. All routing intelligence lives in Q(s,a). This forces the model to learn precise navigation rather than compensating for imprecise attention with a learned content transform.

At H=512, r=64: standard attention uses 1,048,576 parameters per layer. Q-Compass uses 327,680 — a **69% reduction** in attention-block parameters.

The same block — with or without a causal mask — handles both autoregressive text generation (Q-Compass) and bidirectional image encoding (Q-Compass-Bi). One mechanism, all modalities.

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
├── LayerNorm
└── Linear projection → LM hidden dim
```

---

## Modality Support

| Modality | Module | Status |
|----------|--------|--------|
| Text | `QuatrixLM` | Production |
| Vision | `QuatrixVision` | Production |
| Audio | `QuatrixAudio` | Production |
| World Model | `QuatrixWorld` | Production |

---

## Quick Start

```python
from quatrix import QuatrixLM, QuatrixConfig

cfg = QuatrixConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=7,
    max_seq_len=5120,
    q_rank=64,
    use_vision=True,
)
model = QuatrixLM(cfg)  # ~50M params

import torch
input_ids = torch.randint(0, 50257, (1, 10))
out = model(input_ids)
logits = out['logits']  # [B, L, vocab_size]

# Multimodal
pixel_values = torch.randn(1, 3, 224, 224)
out = model(input_ids, pixel_values=pixel_values)
```

---

## Berry-Q0 — First Quatrix Model

**Berry-Q0** is the first model trained on the Quatrix architecture.

| Property | Value |
|----------|-------|
| Architecture | QuatrixLM + QuatrixVision |
| Parameters | ~50M (44M LM + 5.5M Vision + 0.4M projection) |
| Context | 5120 tokens |
| Modalities | Text + Image |
| Training hardware | Single RTX 4050 6GB laptop GPU |
| Text data | ~3.2M samples (web, math, code, reasoning, instruction, alignment) |
| Image data | ~550K image-text pairs (VQAv2, GQA, TextVQA, DocVQA, ScienceQA, CLEVR) |
| Status | GRPO reasoning training in progress |

Trained from scratch in three stages: pretraining on ~3.2M mixed text + image samples, supervised finetuning on instruction and reasoning data, and ongoing GRPO reasoning training (R1-style, math domain). Empirical results will be reported in a follow-up paper once training is complete.

---

## Roadmap

| Model | Modalities | Status |
|-------|-----------|--------|
| Berry-Q0 | Text + Vision | GRPO training in progress |
| Berry-Q1 | Text + Vision + Audio + World Model | Future work |

---

## Paper

If you use Quatrix or Q-Compass in your work, please cite:

```
Syed Abdur Rehman Ali. Q-Compass: Grounding Sequence Mixing in
Reinforcement Learning Navigation. Zenodo, March 2026.
https://zenodo.org/records/19104202
```

---

## Author

**Syed Abdur Rehman Ali**

[![GitHub](https://img.shields.io/badge/GitHub-Abd0r-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abd0r)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Abd0r-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/Abd0r)
[![X](https://img.shields.io/badge/X%20%2F%20Twitter-SyedAbdurR2hman-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/SyedAbdurR2hman)

---

## License

OpenRAIL-M — open use with behavioral restrictions (no military use, no mass surveillance).
See LICENSE for details.
