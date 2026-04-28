"""QuatrixCancerModel — signature decomposition + pan-cancer mutation prediction.

Task: given a tumor mutation catalog (96-dim SBS context vector), predict
      the non-negative signature contribution vector that best reconstructs it.

      In COSMIC terms: observe v ∈ Δ^96, predict c ∈ Δ^K such that v ≈ S·c.

Architecture (the same Q-Compass primitive used in text/vision/audio/world/edit):

    SBS96 mutation catalog (B, 96)
         │
         ▼   EmbedMutationCounts (linear → (B, 96, H))
    context embeddings (B, 96, H)
         │
         ▼   Q-Compass blocks (bidirectional)               — REUSED primitive
    context features (B, 96, H)
         │
         ▼   StateEncoder (learnable query + QCompassBi)     — REUSED
    tumor state (B, H)
         │   + action = cancer-type embedding (optional, for TCGA pan-cancer)
         ▼
    TransitionModel                                          — REUSED
    latent decomposition vector (B, H)
         │
         ▼   Head: linear → softmax over K signatures
    ĉ (B, K)  — predicted signature contributions

Loss: KL(true mixture c || predicted ĉ)  OR
      MSE on reconstructed catalog S·ĉ vs v.

Parameters at default config (H=192, r=24, L=3 blocks) — ~2.5M params.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import QCompassBi, QuatrixVisionBlock
from .model import QCompass   # multi-head + Q-value-content capable


class _ContextEmbedder(nn.Module):
    """Turn a 96-dim mutation count vector into a (96, H) sequence of embeddings.

    Each context has a learned position embedding; the count (or probability)
    is multiplied in as a scalar magnitude, then added to the position embed.
    This is a natural "value + position" encoding for count data.
    """

    def __init__(self, n_contexts: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_contexts = n_contexts
        self.context_emb = nn.Embedding(n_contexts, hidden_size)
        self.count_proj  = nn.Linear(1, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            counts: (B, 96) float — can be raw counts or normalised probabilities.
                    We log1p-transform and project, so the scale is stable.
        Returns:
            (B, 96, H) sequence of per-context tokens.
        """
        B, N = counts.shape
        assert N == self.n_contexts
        ctx_ids = torch.arange(N, device=counts.device).unsqueeze(0).expand(B, N)
        ctx_emb = self.context_emb(ctx_ids)                        # (B, 96, H)
        count_feat = self.count_proj(
            torch.log1p(counts.clamp_min(0)).unsqueeze(-1))         # (B, 96, H)
        return self.drop(ctx_emb + count_feat)


class _CancerStateEncoder(nn.Module):
    """Query-token aggregation (B, N, H) → (B, H)."""

    def __init__(self, hidden_size: int, q_rank: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.compass = QCompassBi(hidden_size, q_rank, dropout)
        self.norm    = nn.LayerNorm(hidden_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B = seq.size(0)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, seq], dim=1)
        x = self.norm(x + self.compass(x))
        return x[:, 0, :]


class _Transition(nn.Module):
    """Transition: state × action → post-edit latent. Uses full MH-QVC blocks."""

    def __init__(self, hidden_size: int, action_dim: int,
                 n_layers: int = 2, q_rank: int = 96,
                 num_heads: int = 6, q_value_content: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.state_proj  = nn.Linear(hidden_size, hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.fusion      = nn.Linear(hidden_size * 2, hidden_size)

        # MH-QVC blocks matching the encoder configuration.
        # Falls back to QuatrixVisionBlock (single-head) if num_heads=1 and
        # q_value_content=False, so backward-compat is preserved.
        if num_heads > 1 or q_value_content:
            self.blocks = nn.ModuleList([
                _MHBlock(hidden_size, q_rank,
                         num_heads=num_heads,
                         q_value_content=q_value_content,
                         ffn_ratio=4, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
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


class _MHBlock(nn.Module):
    """Multi-head + D.1 Q-value-content block for the cancer encoder.

    Uses the same QCompass class as the main LM, so "multi-head" + "Q-value-content"
    behave identically across all Quatrix tasks (text, vision, audio, world,
    gene editing, cancer).
    """

    def __init__(self, hidden_size: int, q_rank: int, num_heads: int = 6,
                 q_value_content: bool = True, ffn_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.compass = QCompass(
            hidden_size, q_rank,
            num_heads=num_heads,
            q_value_content=q_value_content,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        ffn_dim = hidden_size * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use causal=False — cancer context is non-sequential (bidirectional)
        x = x + self.compass(self.norm1(x), causal=False)
        x = x + self.ffn(self.norm2(x))
        return x


class QuatrixCancerModel(nn.Module):
    """Q-Compass cancer model — full-scale, multi-head, D.1 Q-value content enabled.

    Default config is scaled to match the 180m Quatrix LM (H=768, r=96, L=14),
    so the cancer branch uses the SAME maxed-out primitive as the main
    architecture. This makes the "unified primitive at full scale" claim
    concrete: the exact same QCompass module with MH=6 and Q-value content
    on is used across text / vision / audio / world / cancer.
    """

    def __init__(self,
                 n_contexts: int = 96,
                 n_signatures: int = 79,
                 # Defaults: 60m-scale body (right-sized for 96→79 regression
                 # with ~10K real patients / 50K synthetic tumors). Can be
                 # bumped to 180m config via CLI flags for the scaling study.
                 hidden_size: int = 384,      # matches Quatrix-60m
                 q_rank: int = 48,            # matches Quatrix-60m
                 encoder_layers: int = 6,     # deeper than LM-60m (10) is wasted here
                 transition_layers: int = 4,  # small, task isn't long-horizon
                 num_heads: int = 6,          # MH-QVC still on by default
                 q_value_content: bool = True,
                 action_dim: int = 64,
                 num_cancer_types: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.n_contexts = n_contexts
        self.n_signatures = n_signatures
        self.num_cancer_types = num_cancer_types
        self.num_heads = num_heads
        self.q_value_content = q_value_content

        # Enforce compatibility with MH: H and r must be divisible by num_heads
        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert q_rank % num_heads == 0, \
            f"q_rank {q_rank} must be divisible by num_heads {num_heads}"

        self.embed = _ContextEmbedder(n_contexts, hidden_size, dropout)

        # Encoder: full MH-QVC Q-Compass blocks
        self.blocks = nn.ModuleList([
            _MHBlock(hidden_size, q_rank,
                     num_heads=num_heads,
                     q_value_content=q_value_content,
                     ffn_ratio=4, dropout=dropout)
            for _ in range(encoder_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        self.state_enc = _CancerStateEncoder(hidden_size, q_rank, dropout)

        if num_cancer_types > 1:
            self.cancer_emb = nn.Embedding(num_cancer_types, action_dim)
            transition_action_dim = action_dim
        else:
            self.cancer_emb = None
            transition_action_dim = hidden_size

        # Transition: MH-QVC enabled for consistency with encoder
        self.transition = _Transition(
            hidden_size, transition_action_dim,
            n_layers=transition_layers, q_rank=q_rank,
            num_heads=num_heads, q_value_content=q_value_content,
            dropout=dropout)

        self.head = nn.Linear(hidden_size, n_signatures)

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
                counts: torch.Tensor,                          # (B, 96)
                target_contributions: Optional[torch.Tensor] = None,  # (B, K) true c
                signature_matrix: Optional[torch.Tensor] = None,      # (96, K) S
                cancer_type_ids: Optional[torch.Tensor] = None):       # (B,) long
        B = counts.size(0)

        seq = self.embed(counts)                                   # (B, 96, H)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        state = self.state_enc(seq)                                # (B, H)

        # Action:
        if self.cancer_emb is not None and cancer_type_ids is not None:
            action = self.cancer_emb(cancer_type_ids)              # (B, action_dim)
        else:
            action = state                                          # identity action: use state itself

        latent = self.transition(state, action)                    # (B, H)
        logits = self.head(latent)                                  # (B, K)
        pred_c = F.softmax(logits, dim=-1)

        loss = None
        recon_loss = None
        if target_contributions is not None:
            # Cross-entropy between true mixture and predicted mixture
            log_pred = F.log_softmax(logits, dim=-1)
            # target_contributions may be sparse; use it as a soft distribution
            # (already normalised to simplex)
            loss = -(target_contributions * log_pred).sum(dim=-1).mean()

        if signature_matrix is not None:
            # Reconstruction loss: S · ĉ should match observed distribution
            probs_obs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_pred = pred_c @ signature_matrix.T               # (B, 96)
            recon_loss = F.mse_loss(probs_pred, probs_obs)
            loss = recon_loss if loss is None else (loss + 0.5 * recon_loss)

        return {
            "pred_c":      pred_c,
            "logits":      logits,
            "latent":      latent,
            "loss":        loss,
            "recon_loss":  recon_loss,
        }
