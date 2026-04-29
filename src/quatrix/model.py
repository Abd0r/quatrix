"""Quatrix model — Q-Compass language model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .config import QuatrixConfig
from .vision import VisionEncoder
from .audio import AudioEncoder
from .world import WorldModel, WM_ACTION_DIM


class QCompass(nn.Module):
    """
    Q(s,a) Compass — core navigation mechanism.

    Routing:
      Q(s, a) = state(s) · action(a)^T / sqrt(q_rank)
      weights = softmax(Q + causal_mask)

    Content modes:
      q_value_content=False (default, foundation paper):
          output = W_o · (weights @ x)        ← gather raw x
      q_value_content=True (D.1 extension):
          qval    = state ⊙ action            ← elementwise self-Q-value
          content = qval · W_c                ← project Q-value to hidden
          output  = W_o · (weights @ content) ← gather self-Q-values

    Heads:
      num_heads=1 (default): single-head Q-Compass.
      num_heads>1: multi-head — q_rank split across heads (each head has
                   r/h rank). Routing is per-head; content slicing is per-head.

    Key difference from standard attention: no W_v projecting raw x.
    Either we gather raw x directly, or we gather the token's OWN
    self-Q-value (computed from the same state/action projections).
    """

    def __init__(self, hidden_size: int, q_rank: int,
                 num_heads: int = 1,
                 q_value_content: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        assert q_rank % num_heads == 0, \
            f"q_rank {q_rank} must be divisible by num_heads {num_heads}"
        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"

        self.hidden_size     = hidden_size
        self.q_rank          = q_rank
        self.num_heads       = num_heads
        self.head_r          = q_rank // num_heads
        self.head_H          = hidden_size // num_heads
        self.q_value_content = q_value_content
        self.scale           = math.sqrt(self.head_r)

        self.state_proj  = nn.Linear(hidden_size, q_rank, bias=False)
        self.action_proj = nn.Linear(hidden_size, q_rank, bias=False)
        self.out_proj    = nn.Linear(hidden_size, hidden_size)
        self.drop        = nn.Dropout(dropout)

        # D.1 content projection: project self-Q-value (rank r) to hidden H.
        # Only allocated if enabled; preserves the foundation "no W_v" property
        # when disabled.
        if q_value_content:
            self.content_proj = nn.Linear(q_rank, hidden_size, bias=False)
        else:
            self.content_proj = None

        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (self._causal_mask is None
                or self._causal_mask.size(0) < seq_len
                or self._causal_mask.device != device):
            self._causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, L, H = x.shape
        h = self.num_heads

        # Fast path for single-head standard Q-Compass (backward compat)
        if h == 1 and not self.q_value_content:
            state  = self.state_proj(x)
            action = self.action_proj(x)
            q_values = torch.bmm(state, action.transpose(-2, -1)) / self.scale
            if causal:
                mask = self._get_causal_mask(L, x.device)
                q_values = q_values.masked_fill(mask == 0, torch.finfo(q_values.dtype).min)
            weights = F.softmax(q_values, dim=-1)
            weights = self.drop(weights)
            return self.out_proj(torch.bmm(weights, x))

        # Multi-head and/or D.1 path
        state  = self.state_proj(x).view(B, L, h, self.head_r).transpose(1, 2)   # [B, h, L, r/h]
        action = self.action_proj(x).view(B, L, h, self.head_r).transpose(1, 2)  # [B, h, L, r/h]

        # Per-head attention scores
        q_values = torch.matmul(state, action.transpose(-2, -1)) / self.scale    # [B, h, L, L]
        if causal:
            mask = self._get_causal_mask(L, x.device)
            q_values = q_values.masked_fill(mask == 0, torch.finfo(q_values.dtype).min)
        weights = F.softmax(q_values, dim=-1)
        weights = self.drop(weights)

        if self.q_value_content:
            # D.1: content is self-Q-value (state ⊙ action), projected to H
            qval     = state * action                                # [B, h, L, r/h]
            qval_flat = qval.transpose(1, 2).reshape(B, L, self.q_rank)   # [B, L, r]
            content   = self.content_proj(qval_flat)                 # [B, L, H]
            # Split content across heads for per-head gather
            content_h = content.view(B, L, h, self.head_H).transpose(1, 2)   # [B, h, L, H/h]
            gathered  = torch.matmul(weights, content_h)             # [B, h, L, H/h]
        else:
            # Multi-head standard Q-Compass: per-head gather from x slice
            x_h      = x.view(B, L, h, self.head_H).transpose(1, 2)  # [B, h, L, H/h]
            gathered = torch.matmul(weights, x_h)                    # [B, h, L, H/h]

        # Concatenate heads back to [B, L, H]
        out = gathered.transpose(1, 2).reshape(B, L, H)
        return self.out_proj(out)


class QuatrixBlock(nn.Module):
    """Pre-norm residual block: QCompass + FFN."""

    def __init__(self, cfg: QuatrixConfig):
        super().__init__()
        self.norm1   = nn.LayerNorm(cfg.hidden_size)
        self.compass = QCompass(
            cfg.hidden_size,
            cfg.q_rank,
            num_heads=getattr(cfg, "q_heads", 1),
            q_value_content=getattr(cfg, "q_value_content", False),
            dropout=cfg.dropout,
        )
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

        # Vision encoder (optional)
        self.vision: Optional[VisionEncoder] = None
        if cfg.use_vision:
            self.vision = VisionEncoder(cfg.hidden_size, cfg.dropout)

        # Audio encoder (optional) — prepended before text tokens, after vision
        self.audio: Optional[AudioEncoder] = None
        if cfg.use_audio:
            self.audio = AudioEncoder(cfg.hidden_size, dropout=cfg.dropout)

        # World model (optional) — state-action-transition plugin
        self.world: Optional[WorldModel] = None
        self.world_action_emb: Optional[nn.Embedding] = None
        if cfg.use_world_model:
            wm_layers = cfg.wm_layers if cfg.wm_layers > 0 else cfg.num_layers
            self.world = WorldModel(
                lm_hidden=cfg.hidden_size,
                n_transition_layers=wm_layers,
                q_rank=cfg.q_rank,
            )
            self.world_action_emb = nn.Embedding(cfg.world_num_actions, WM_ACTION_DIM)

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
        mel_values: Optional[torch.Tensor] = None,
        world_batch: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            input_ids:    [B, L]
            labels:       [B, L]  (optional, for LM loss)
            pixel_values: [B, 3, 224, 224]  (optional vision tokens)
            mel_values:   [B, 1, 80, T]     (optional audio tokens)
            world_batch:  dict with keys:
                            "frame_t1":   [B, 3, H, W] — next-frame target
                            "action_ids": [B]           — discrete action taken
        """
        B, L = input_ids.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}. "
                f"Truncate input_ids before calling forward()."
            )
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_emb(input_ids) + self.pos_emb(pos)   # [B, L, H]

        # --- Collect prefix tokens (vision first, then audio) ---
        prefix_parts = []
        vis_emb = None   # saved for world model reuse

        if pixel_values is not None and self.vision is not None:
            vis_emb = self.vision(pixel_values)        # [B, N_vis, H]
            prefix_parts.append(vis_emb)

        if mel_values is not None and self.audio is not None:
            aud_emb = self.audio(mel_values)           # [B, N_aud, H]
            prefix_parts.append(aud_emb)

        if prefix_parts:
            prefix = torch.cat(prefix_parts, dim=1)
            prefix_len = prefix.size(1)
            x = self.drop(torch.cat([prefix, tok_emb], dim=1))
        else:
            prefix_len = 0
            x = self.drop(tok_emb)

        for block in self.blocks:
            if self.cfg.use_gradient_checkpointing and self.training:
                x = _ckpt(lambda _x, _b=block: _b(_x, causal=causal), x, use_reentrant=False)
            else:
                x = block(x, causal=causal)

        x = self.norm(x)
        logits = self.head(x)   # [B, prefix_len+L, vocab]

        loss = None
        if labels is not None:
            text_logits = logits[:, prefix_len:, :]        # [B, L, vocab]
            shift_logits = text_logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.cfg.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        # --- World model: predict next visual state from (state_t, action) ---
        world_loss = None
        if (world_batch is not None and self.world is not None
                and self.world_action_emb is not None
                and vis_emb is not None):
            frame_t1   = world_batch["frame_t1"].to(x.device)    # [B, 3, H, W]
            action_ids = world_batch["action_ids"].to(x.device)  # [B]

            state_t    = self.world.state_encoder(vis_emb)              # [B, H]
            action_emb = self.world_action_emb(action_ids)             # [B, WM_ACTION_DIM]
            pred_next  = self.world.transition(state_t, action_emb)    # [B, H]

            with torch.no_grad():
                vis_t1   = self.vision(frame_t1)                       # [B, N_vis, H]
                state_t1 = self.world.state_encoder(vis_t1)            # [B, H]

            world_loss = F.mse_loss(pred_next, state_t1)

            # World MSE is smaller-magnitude than text CE; weight 0.5 balances them
            if loss is not None:
                loss = loss + 0.5 * world_loss
            else:
                loss = world_loss

        return {'loss': loss, 'logits': logits, 'world_loss': world_loss}
