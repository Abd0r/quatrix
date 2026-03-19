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
| Parameters | ~50M (44M LM + 5.5M Vision + 0.4M projection) |
| Context | 5120 tokens |
| Modalities | Text + Image (VLM) |
| Training | From scratch, single RTX 4050 6GB |
| Data | ~3.2M text samples + ~550K image-text pairs |
| Status | **GRPO reasoning training in progress (inspired by DeepSeek R1)** |

A proof-of-concept (earlier smaller run — 1M diverse samples, 3 pretraining epochs, 256 context limit) scored **48.8% HumanEval Pass@1** — matching GPT-3.5 (175B) at **3500× fewer parameters**.

---

## Berry-Q0 Training Journey

### Stage 1 — Pretraining (`train_berry_multimodal.py`)

3 epochs of progressive context training on ~3.2M mixed text + image samples:

| Epoch | Context | Batch | Purpose |
|-------|---------|-------|---------|
| 1 | 256 tokens | 26 | Fast warmup, short patterns |
| 2 | 1024 tokens | 5 | Medium context |
| 3 | 5120 tokens | 1 | Full 5K context |

**Data:** FineWeb-Edu 500K, C4 300K, NuminaMath+OpenR1+OpenMathInstruct ~600K, Magicoder 75K, GitHub code 300K, SmolTalk 300K, OASST2 100K, hh-rlhf 100K, ArXiv 197K, VQAv2, GQA, TextVQA, DocVQA, ScienceQA, and more.

**Optimizer:** Muon (21.2M params, lr=3e-3) + AdamW (28.8M params, lr=3e-4), cosine decay, warmup 500 steps.

### Stage 2 — SFT (`train_berry_multimodal.py`)

6 epochs of supervised finetuning on curated instruction + reasoning data:

- **Epochs 1–4:** 512-token context, 150K samples/epoch (instruction + reasoning mix)
- **Epochs 5–6:** 5120-token long context, 50K samples/epoch

**Categories:** SmolTalk, OASST2, hh-rlhf, GPT4-LLM, OpenHermes, CodeAlpaca, CodeFeedback, Glaive function-calling, HelpSteer, Prosocial, DEITA, OpenR1, OpenThoughts, NuminaMath, GSM8K, ARC, HellaSwag, TriviaQA, and more.

**LR:** 1.5e-3 / 1.5e-4 (2x lower than pretraining). Final checkpoint: `checkpoint-epsft6-step46003`.

### Stage 3 — GRPO (`train_berry_grpo.py`) — In Progress

**R1-style reasoning training** — Math + Code only, pure ground truth binary reward. Inspired by DeepSeek R1's approach: no complex reward stacks, no hackable heuristics. Only verifiable correctness.

| Domain | Samples | Reward |
|--------|---------|--------|
| Math | 40K | Exact number match (binary 0/1) |
| Code | 35K | Execution pass rate (sandboxed subprocess) |

**Config:**
- G=8 completions/prompt, 25K steps, 512 max tokens
- KL penalty β=0.001 (R1 exact value), grad_accum=2
- Temperature=1.0 (R1 style — full exploration)
- Vision params frozen (4.6M) — language-only training

**Reward function (`rewards/accuracy_reward.py`):**
```
fmt_reward  = 1.0 if reasoning pattern + answer pattern present, else 0.5/0.0
acc_reward  = 1.0 if correct answer, else 0.0
total       = (fmt + acc) / 2  →  [0.0, 0.5, 1.0]
```
- Math: exact number extraction + match (`\boxed{}`, `#### X`, `the answer is X`)
- Code: sandboxed subprocess execution, recursionlimit=1000
- Diversity penalty: rewards × 0.05 if all G=8 completions near-identical (exploit detection)

**Prompt format:** `user: [problem]\nassistant: <think>\n` — triggers model's trained CoT format learned during SFT on OpenThoughts/OpenR1 data.

**Advantage computation:** Standard within-group normalization (R1 style) — no rolling baseline. `adv = (r - mean_G) / (std_G + ε)`. Prevents collapse from stale historical baselines.

**Optimizer:** Muon (2.5e-4) + AdamW (2.5e-5) — 6x lower than SFT LR, warmup 100 steps.

---

## Quasar Series Roadmap

| Model | Params | Modalities | Status |
|-------|--------|-----------|--------|
| Berry-Q0 | 50M | Text + Vision | GRPO reasoning training in progress (inspired by DeepSeek R1) |
| Berry-Q1 | ~100M | Text + Vision + Audio + World Model | 📋 Planned — target: beat R1 across benchmarks |
| Berry-Q2 | ~500M | Full multimodal + Robotics + Agents | 📋 Planned |

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
