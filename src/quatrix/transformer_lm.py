"""Standard Transformer LM baseline — matched to QuatrixLM except the attention kernel.

Same embedding dim, layer count, FFN ratio, dropout, tied embeddings, and init as
QuatrixLM. The only difference: Q-Compass (3 projections: W_s, W_a, W_o) is
swapped for standard multi-head attention (4 projections: W_Q, W_K, W_V, W_O).

Used only as a text-only baseline for the paper.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .config import QuatrixConfig


class MHA(nn.Module):
    """Rank-matched single-head attention — all 3 learned projections at rank r.

    Apples-to-apples ablation vs Q-Compass:
      Q-Compass has TWO learned projections (state, action) at rank r.
      Transformer has THREE learned projections (Q, K, V) at rank r.
      The question under test: does an extra learned rank-r projection for
      content (V) beat Q-Compass's "gather raw x directly" approach?

      - W_Q: H × r
      - W_K: H × r
      - W_V: H × r   (same rank as Q and K)
      - W_O: r × H   (projects rank-r content back to H)

    Scores = softmax(Q·K^T / sqrt(r))  →  [L, L]
    Output = W_O · (scores @ V)        →  [L, H]
    """

    def __init__(self, hidden_size: int, qk_rank: int,
                 dropout: float = 0.1, num_heads: int = 1):
        super().__init__()
        assert qk_rank % num_heads == 0, \
            f"qk_rank {qk_rank} must be divisible by num_heads {num_heads}"
        self.qk_rank   = qk_rank
        self.num_heads = num_heads
        self.head_r    = qk_rank // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_r)

        self.q_proj = nn.Linear(hidden_size, qk_rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, qk_rank, bias=False)   # rank-r (matched)
        self.o_proj = nn.Linear(qk_rank,  hidden_size)              # r → H
        self.drop   = nn.Dropout(dropout)
        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        if (self._causal_mask is None
                or self._causal_mask.size(0) < L
                or self._causal_mask.device != device):
            self._causal_mask = torch.tril(torch.ones(L, L, device=device))
        return self._causal_mask[:L, :L]

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, L, _ = x.shape
        h, r_h  = self.num_heads, self.head_r

        # Single-head fast path (backward compat with previous checkpoints).
        if h == 1:
            q = self.q_proj(x)                       # [B, L, r]
            k = self.k_proj(x)
            v = self.v_proj(x)
            scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale
            if causal:
                mask = self._get_causal_mask(L, x.device)
                scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = self.drop(weights)
            return self.o_proj(torch.bmm(weights, v))

        # Multi-head: split r across heads. Total projection params unchanged.
        q = self.q_proj(x).view(B, L, h, r_h).transpose(1, 2)    # [B, h, L, r_h]
        k = self.k_proj(x).view(B, L, h, r_h).transpose(1, 2)
        v = self.v_proj(x).view(B, L, h, r_h).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, L, L]
        if causal:
            mask = self._get_causal_mask(L, x.device)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out = torch.matmul(weights, v)                              # [B, h, L, r_h]
        out = out.transpose(1, 2).contiguous().view(B, L, h * r_h)  # [B, L, r]
        return self.o_proj(out)                                     # [B, L, H]


class TransformerBlock(nn.Module):
    """Pre-norm residual block: MHA + FFN. Mirrors QuatrixBlock structure."""

    def __init__(self, cfg: QuatrixConfig, qk_rank: int = 48, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn  = MHA(cfg.hidden_size, qk_rank, cfg.dropout, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        ffn_dim = cfg.hidden_size * cfg.ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Standard transformer language model — matched baseline to QuatrixLM.

    Identical to QuatrixLM in: embedding dim, layer count, FFN ratio, dropout,
    tied embeddings, init distribution, optimizer routing.
    Different: attention kernel uses W_Q, W_K, W_V, W_O instead of W_s, W_a, W_o.
    """

    def __init__(self, cfg: QuatrixConfig, qk_rank: int = 48, num_heads: int = 1):
        super().__init__()
        self.cfg = cfg
        self.qk_rank = qk_rank
        self.num_heads = num_heads
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(cfg, qk_rank=qk_rank, num_heads=num_heads) for _ in range(cfg.num_layers)
        ])
        self.norm      = nn.LayerNorm(cfg.hidden_size)
        self.head      = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        causal: bool = True,
        **kwargs,
    ):
        B, L = input_ids.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}. "
                f"Truncate input_ids before calling forward()."
            )
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            if self.cfg.use_gradient_checkpointing and self.training:
                x = _ckpt(lambda _x, _b=block: _b(_x, causal=causal), x, use_reentrant=False)
            else:
                x = block(x, causal=causal)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.cfg.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "world_loss": None}
