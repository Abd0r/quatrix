"""QuatrixEditModel — gene-editing outcome predictor built on Q-Compass.

Architectural mirror of the generative world model:

    target DNA + editor + <cut>   (L tokens)
           │
           ▼   DNA encoder: bidirectional QCompassBi blocks                 — REUSED primitive
    context embeddings (L, H)
           │
           ▼   StateEncoder (QCompassBi + query token)                      — REUSED
    state_t  (B, H)                    ← "what the model sees"
           │      ┌── action = editor-type embedding                        — small adapter
           ▼      ▼
    TransitionModel (QCompassBi blocks)                                    — REUSED
    state_t+1  (B, H)                  ← "post-edit latent"
           │
           ▼   Outcome head: dot-product against per-candidate outcome embeddings
    logits over K candidate outcomes (B, K)
           │
           ▼   softmax = predicted edit-outcome distribution
    cross-entropy loss against empirical distribution

KEY DESIGN CHOICE:
  We don't regenerate the full edited DNA sequence. Instead, we use an
  OUTCOME-SCORING formulation (like ranking / contrastive):
    1. Encode each of the K candidate outcomes with the SAME DNA encoder.
    2. Aggregate each outcome to a single vector via StateEncoder.
    3. Compute score = <state_t+1, outcome_k> for each k.
    4. Softmax over K gives predicted distribution.

  This matches how inDelphi and FORECasT report: top-K outcome distributions
  rather than the one "most likely" edited sequence. It also sidesteps the
  harder problem of autoregressive DNA generation, letting us benchmark
  directly against published metrics (KL, Pearson, accuracy@1).

Parameter budget (default: H=192, r=24, L=4 blocks, base tokenizer):
  embeddings      : ~0.03M  (vocab=15-ish)
  DNA encoder     : ~1.5M   (4 QCompassBi)
  StateEncoder    : ~0.12M  (1 QCompassBi)
  TransitionModel : ~1.8M   (4 QCompassBi)
  Outcome scoring : ~0.04M  (small MLP)
  Total           : ~3.5M   — SOTA comparisons typically have 1-10M params
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import QCompassBi, QuatrixVisionBlock


class _DNAEncoder(nn.Module):
    """Bidirectional Q-Compass on DNA tokens."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 n_layers: int = 4, q_rank: int = 24,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size,
                                       padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len, hidden_size)
        self.drop      = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(hidden_size, q_rank, ffn_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class _StateEncoder(nn.Module):
    """Compress a token sequence (B, L, H) → state vector (B, H) via query token."""

    def __init__(self, hidden_size: int, q_rank: int, dropout: float = 0.1):
        super().__init__()
        self.query   = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.compass = QCompassBi(hidden_size, q_rank, dropout)
        self.norm    = nn.LayerNorm(hidden_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B = seq.size(0)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, seq], dim=1)
        x = self.norm(x + self.compass(x))
        return x[:, 0, :]


class _TransitionModel(nn.Module):
    """state × editor-action → next-state (post-edit latent)."""

    def __init__(self, hidden_size: int, action_dim: int,
                 n_layers: int = 4, q_rank: int = 24, dropout: float = 0.1):
        super().__init__()
        self.state_proj  = nn.Linear(hidden_size, hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.fusion      = nn.Linear(hidden_size * 2, hidden_size)
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(hidden_size, q_rank, ffn_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.out  = nn.Linear(hidden_size, hidden_size)

    def forward(self, state: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        s = self.state_proj(state)
        a = self.action_proj(action_emb)
        x = self.fusion(torch.cat([s, a], dim=-1)).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        return self.out(self.norm(x).squeeze(1))


class QuatrixEditModel(nn.Module):
    """Gene-editing outcome predictor, Q-Compass throughout."""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 192,
                 q_rank: int = 24,
                 encoder_layers: int = 4,
                 transition_layers: int = 4,
                 action_dim: int = 32,
                 num_editors: int = 5,           # Cas9 Cas12a ABE CBE pegRNA
                 max_target_len: int = 128,
                 max_outcome_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Shared DNA encoder for target AND candidate outcomes
        self.dna_encoder = _DNAEncoder(vocab_size, hidden_size,
                                       n_layers=encoder_layers, q_rank=q_rank,
                                       max_len=max(max_target_len, max_outcome_len),
                                       dropout=dropout)

        # State compressors (separate for target / outcome, but same primitive)
        self.state_enc   = _StateEncoder(hidden_size, q_rank, dropout)
        self.outcome_enc = _StateEncoder(hidden_size, q_rank, dropout)

        # Editor-type embedding (the "action" for this world model)
        self.editor_emb  = nn.Embedding(num_editors, action_dim)

        # Transition model: state_t + editor → predicted post-edit state_t+1
        self.transition  = _TransitionModel(hidden_size, action_dim,
                                            n_layers=transition_layers,
                                            q_rank=q_rank, dropout=dropout)

        # Scoring: temperature on the dot-product between predicted state_t+1
        # and each candidate outcome's state vector
        self.log_temperature = nn.Parameter(torch.zeros(1))

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

    def forward(self,
                target_ids: torch.Tensor,           # (B, L)
                outcome_ids: torch.Tensor,          # (B, K, L_out)
                outcome_freqs: Optional[torch.Tensor] = None,   # (B, K) empirical dist
                editor_ids: Optional[torch.Tensor] = None):     # (B,) long
        B, L = target_ids.shape
        K, L_out = outcome_ids.size(1), outcome_ids.size(2)

        # Encode target sequence → state_t
        tgt_emb = self.dna_encoder(target_ids)                 # (B, L, H)
        state_t = self.state_enc(tgt_emb)                      # (B, H)

        # Action: editor-type. Fall back to "Cas9" (id 0) if not given.
        if editor_ids is None:
            editor_ids = torch.zeros(B, dtype=torch.long, device=target_ids.device)
        action = self.editor_emb(editor_ids)                   # (B, action_dim)

        # Predicted post-edit state
        pred_state = self.transition(state_t, action)          # (B, H)

        # Encode each of the K candidate outcomes
        out_flat = outcome_ids.view(B * K, L_out)
        out_emb  = self.dna_encoder(out_flat)                  # (B*K, L_out, H)
        out_state = self.outcome_enc(out_emb).view(B, K, -1)   # (B, K, H)

        # Scores = <pred_state, outcome_k> / temperature
        temp = self.log_temperature.exp()
        logits = torch.einsum("bh,bkh->bk", pred_state, out_state) / temp

        loss = None
        if outcome_freqs is not None:
            # Empirical distribution is already normalised (K-simplex).
            # Cross entropy between predicted (softmax of logits) and target distribution.
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(outcome_freqs * log_probs).sum(dim=-1).mean()

        return {
            "logits":     logits,
            "pred_state": pred_state,
            "out_state":  out_state,
            "loss":       loss,
        }

    @torch.no_grad()
    def predict_distribution(self, target_ids, outcome_ids, editor_ids=None):
        out = self.forward(target_ids, outcome_ids, editor_ids=editor_ids)
        return torch.softmax(out["logits"], dim=-1)
