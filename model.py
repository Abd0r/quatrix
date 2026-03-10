"""Quatrix model — Q-Compass language model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .config import QuatrixConfig
from .vision import VisionEncoder


class QCompass(nn.Module):
    """
    Q(s,a) Compass — core navigation mechanism.

    Q(s, a) = state(s) · action(a)^T / sqrt(q_rank)
    weights  = softmax(Q + causal_mask)
    output   = weights @ x        ← NO value projection

    Key difference from standard attention: the model learns
    WHICH positions to mix; the content being mixed is x itself.
    """

    def __init__(self, hidden_size: int, q_rank: int, dropout: float = 0.1):
        super().__init__()
        self.state_proj  = nn.Linear(hidden_size, q_rank, bias=False)
        self.action_proj = nn.Linear(hidden_size, q_rank, bias=False)
        self.out_proj    = nn.Linear(hidden_size, hidden_size)
        self.drop        = nn.Dropout(dropout)
        self.scale       = math.sqrt(q_rank)
        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) < seq_len:
            self._causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, L, H = x.shape

        state  = self.state_proj(x)   # [B, L, q_rank]
        action = self.action_proj(x)  # [B, L, q_rank]

        q_values = torch.bmm(state, action.transpose(-2, -1)) / self.scale  # [B, L, L]

        if causal:
            mask = self._get_causal_mask(L, x.device)
            q_values = q_values.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(q_values, dim=-1)
        weights = self.drop(weights)

        out = torch.bmm(weights, x)   # [B, L, H] — gather directly from x
        return self.out_proj(out)


class QuatrixBlock(nn.Module):
    """Pre-norm residual block: QCompass + FFN."""

    def __init__(self, cfg: QuatrixConfig):
        super().__init__()
        self.norm1   = nn.LayerNorm(cfg.hidden_size)
        self.compass = QCompass(cfg.hidden_size, cfg.q_rank, cfg.dropout)
        self.norm2   = nn.LayerNorm(cfg.hidden_size)
        ffn_dim = cfg.hidden_size * cfg.ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.compass(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class QuatrixLM(nn.Module):
    """
    Quatrix language model.

    Returns {'loss': tensor, 'logits': tensor}.
    Accepts causal/use_memory/use_thinking kwargs (extras ignored)
    for drop-in compatibility with the training pipeline.
    """

    def __init__(self, cfg: QuatrixConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([QuatrixBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm      = nn.LayerNorm(cfg.hidden_size)
        self.head      = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.token_emb.weight

        # Vision encoder (optional) — QuatrixVision, no external dependencies
        self.vision: Optional[VisionEncoder] = None
        if cfg.use_vision:
            self.vision = VisionEncoder(cfg.hidden_size, cfg.dropout)

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
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_emb(input_ids) + self.pos_emb(pos)   # [B, L, H]

        # --- Multimodal: prepend image tokens ---
        img_len = 0
        if pixel_values is not None and self.vision is not None:
            img_emb = self.vision(pixel_values)   # [B, 257, H]
            img_len = img_emb.size(1)
            x = self.drop(torch.cat([img_emb, tok_emb], dim=1))   # [B, 257+L, H]
        else:
            x = self.drop(tok_emb)                                 # [B, L, H]

        for block in self.blocks:
            if self.cfg.use_gradient_checkpointing and self.training:
                x = _ckpt(lambda _x, _b=block: _b(_x, causal=causal), x, use_reentrant=False)
            else:
                x = block(x, causal=causal)

        x = self.norm(x)
        logits = self.head(x)   # [B, img_len+L, vocab]

        loss = None
        if labels is not None:
            # Shift within the text portion only (skip image tokens)
            text_logits = logits[:, img_len:, :]           # [B, L, vocab]
            shift_logits = text_logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.cfg.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        return {'loss': loss, 'logits': logits}
