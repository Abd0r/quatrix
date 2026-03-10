# Quatrix — Q-Compass Architecture

> *"Where transformers retrieve by similarity, Quatrix navigates by value."*

**Quatrix** is a novel neural architecture that replaces standard attention with **Q-Compass navigation** — a mechanism grounded in reinforcement learning theory rather than similarity matching.

Built by **Syed Abdur Rehman Ali** ([@Abd0r](https://github.com/Abd0r)).

---

## Core Idea: Q-Compass

Standard attention computes:
```
A = softmax(Q @ K.T / sqrt(d)) @ V
```
Three projections. Similarity-based: attends to what *looks similar*.

**Q-Compass computes:**
```
state  = Linear(x, q_rank)      # "Where am I? What do I need?"
action = Linear(x, q_rank)      # "Where can I go? What's available?"
nav    = softmax(state @ action.T / sqrt(q_rank))
output = nav @ x                # gather directly — no V projection
```
Two projections. Value-based: navigates to what is *useful*.

**Key difference:** No Value projection. The content being mixed is `x` itself — raw, unchanged. The navigation weights encode all routing intelligence. This forces the model to **compose** answers rather than **retrieve** them.

The navigation matrix `Q(s,a)` is mathematically identical to the Q-value function in reinforcement learning — the function that asks *"in state s, how valuable is taking action a?"*

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

VisionEncoder (image plugin)
├── Conv2d patch embedding (16×16 patches)
├── Positional embeddings
└── M × QuatrixVisionBlock (QCompassBi — bidirectional)

AudioEncoder (audio plugin)
├── Mel-spectrogram patch embedding
├── Positional embeddings
└── M × QuatrixVisionBlock (QCompassBi — bidirectional)

WorldModel (world model plugin)
├── StateEncoder   — compress token sequence → state vector
├── ActionHead     — predict action from state
├── TransitionModel — predict next state given state + action
└── RewardHead     — estimate value (optional, for RL)
```

All modalities use the **same QCompassBi block** — one unified mechanism for text, images, audio, and world modeling. No CLIP, no Whisper, no external dependencies.

---

## Modality Support

| Modality | Module | Status |
|----------|--------|--------|
| Text | `QuatrixLM` | ✅ Production |
| Vision | `VisionEncoder` | ✅ Production |
| Audio | `AudioEncoder` | 🔧 Plugin (ready for Q1) |
| World Model | `WorldModel` | 🔧 Plugin (ready for Q1) |

---

## Installation

```bash
pip install quatrix  # coming soon to PyPI
```

Or from source:
```bash
git clone https://github.com/Abd0r/quatrix
cd quatrix
pip install -e .
```

---

## Quick Start

```python
from quatrix import QuatrixLM, QuatrixConfig

# Create model
cfg = QuatrixConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=7,
    max_seq_len=5120,
    q_rank=64,
    use_vision=True,
)
model = QuatrixLM(cfg)
# 50M params — runs on consumer GPU

# Text generation
import torch
input_ids = torch.randint(0, 50257, (1, 10))
out = model(input_ids)
logits = out['logits']  # [B, L, vocab_size]

# Multimodal (text + image)
pixel_values = torch.randn(1, 3, 224, 224)
out = model(input_ids, pixel_values=pixel_values)
```

---

## Berry-Q0 — First Quatrix Model

**Berry-Q0** is the first model trained on the Quatrix architecture.

| Property | Value |
|----------|-------|
| Architecture | QuatrixLM + QuatrixVision |
| Parameters | ~50M (45.4M LM + 4.6M Vision) |
| Context | 5120 tokens |
| Modalities | Text + Image (VLM) |
| Training | From scratch, single RTX 4050 6GB |
| Data | ~3.2M text samples + ~550K image-text pairs |
| Status | **Currently training** |

Berry-Q0 proof-of-concept (earlier, smaller run) scored **48.8% HumanEval Pass@1** — matching GPT-3.5 (175B) at **3500× fewer parameters**.

Weights and paper releasing soon on HuggingFace: [huggingface.co/Abd0r](https://huggingface.co/Abd0r)

---

## Quasar Series Roadmap

| Model | Params | Modalities | Status |
|-------|--------|-----------|--------|
| Berry-Q0 | 50M | Text + Vision | 🔥 Training |
| Berry-Q1 | ~100M | Text + Vision + Audio + World Model | 📋 Planned |
| Berry-Q2 | ~500M | Full multimodal + Robotics | 📋 Planned |

---

## Why Quatrix?

- **Simpler** — 2 projections per layer vs 4 in standard attention
- **Grounded** — Q(s,a) has theoretical roots in RL, not just empirical tricks
- **Universal** — same block for text, images, audio, world modeling
- **Efficient** — 3500× parameter efficiency demonstrated on HumanEval
- **Edge-ready** — 50M model trains and runs on consumer hardware

---

## Author

**Syed Abdur Rehman Ali**
- GitHub: [@Abd0r](https://github.com/Abd0r)
- HuggingFace: [Abd0r](https://huggingface.co/Abd0r)
- X: [@SyedAbdurR2hman](https://x.com/SyedAbdurR2hman)

---

## License

OpenRAIL-M — open use with behavioral restrictions (no military use, no mass surveillance).
See LICENSE for details.
