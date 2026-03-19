#!/usr/bin/env python3
"""
train.py — Minimal QuatrixLM training script.

Downloads TinyShakespeare and trains a small Q-Compass language model.
Runs on CPU or GPU. No setup beyond pip install.

Usage:
  pip install torch transformers
  python train.py                          # quick demo (256 hidden, 4 layers)
  python train.py --steps 2000             # longer run
  python train.py --hidden 512 --layers 7  # Berry-Q0 size (needs GPU)
  python train.py --data myfile.txt        # your own text
"""

import os
import math
import time
import argparse
import urllib.request

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from quatrix import QuatrixLM, QuatrixConfig


# ── Data ──────────────────────────────────────────────────────────────────────

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"Downloading TinyShakespeare → {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
        print("  Done.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TextDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int):
        self.ids     = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int = 200,
             temp: float = 0.8, device: str = "cpu") -> str:
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new):
        if ids.size(1) >= model.cfg.max_seq_len:
            ids = ids[:, -model.cfg.max_seq_len:]
        logits = model(input_ids=ids, causal=True)["logits"][:, -1, :] / temp
        probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, 1)
        if next_t.item() == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, next_t], dim=1)

    model.train()
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


# ── LR schedule ───────────────────────────────────────────────────────────────

def make_lr_fn(warmup: int, total: int):
    def lr_fn(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train QuatrixLM")
    parser.add_argument("--steps",        type=int,   default=1000,        help="total training steps")
    parser.add_argument("--hidden",       type=int,   default=256,         help="hidden size")
    parser.add_argument("--layers",       type=int,   default=4,           help="number of layers")
    parser.add_argument("--q_rank",       type=int,   default=32,          help="Q-Compass rank")
    parser.add_argument("--seq_len",      type=int,   default=256,         help="context length")
    parser.add_argument("--batch",        type=int,   default=8,           help="batch size")
    parser.add_argument("--lr",           type=float, default=3e-4,        help="peak learning rate")
    parser.add_argument("--data",         type=str,   default="input.txt", help="path to .txt file")
    parser.add_argument("--sample_every", type=int,   default=200,         help="generate a sample every N steps")
    parser.add_argument("--save",         type=str,   default="quatrix_checkpoint.pt", help="checkpoint path")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode (training will be slow for large configs)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    text      = get_text(args.data)
    token_ids = tokenizer.encode(text)
    print(f"Data: {len(text):,} chars → {len(token_ids):,} tokens")

    dataset = TextDataset(token_ids, args.seq_len)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset):,} sequences of length {args.seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = QuatrixConfig(
        vocab_size  = tokenizer.vocab_size,
        hidden_size = args.hidden,
        num_layers  = args.layers,
        q_rank      = args.q_rank,
        max_seq_len = args.seq_len * 4,   # headroom for generation
        dropout     = 0.1,
        tie_embeddings = True,
    )
    model   = QuatrixLM(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {n_param:.1f}M params")
    print(f"  hidden={args.hidden}  layers={args.layers}  q_rank={args.q_rank}")
    print(f"  Q-Compass projections: {args.hidden}×{args.q_rank} state + {args.hidden}×{args.q_rank} action  (no V)")

    # ── Optimizer + schedule ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup    = min(100, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_fn(warmup, args.steps))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.steps} steps  (warmup={warmup})\n" + "─" * 50)

    model.train()
    data_iter  = iter(loader)
    step       = 0
    loss_acc   = 0.0
    t0         = time.time()

    while step < args.steps:
        # Refill iterator when dataset exhausted
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y     = next(data_iter)

        x, y = x.to(device), y.to(device)

        out  = model(input_ids=x, labels=y, causal=True)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_acc += loss.item()
        step     += 1

        # Logging
        if step % 10 == 0:
            avg_loss = loss_acc / 10
            lr_now   = optimizer.param_groups[0]["lr"]
            elapsed  = time.time() - t0
            print(f"[{step:5d}/{args.steps}]  loss={avg_loss:.4f}  lr={lr_now:.2e}  {elapsed:.1f}s")
            loss_acc = 0.0
            t0       = time.time()

        # Sample
        if step % args.sample_every == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_new=150, temp=0.8, device=device)
            print(f"\n── Sample (step {step}) ──────────────────────────")
            print(sample[:400])
            print("─" * 50 + "\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           cfg,
        "step":             step,
        "vocab_size":       tokenizer.vocab_size,
    }, args.save)
    print(f"\nCheckpoint saved → {args.save}")

    # Final sample
    print("\n── Final sample ─────────────────────────────────")
    print(generate(model, tokenizer, "ROMEO:", max_new=200, temp=0.8, device=device))
    print("─" * 50)


if __name__ == "__main__":
    main()
