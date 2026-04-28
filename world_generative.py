"""QuatrixWorldGenerative — generative world model built entirely from Q-Compass.

Given a frame and an action, predicts the NEXT FRAME (not just a latent state).
All four sub-modules use the same Q-Compass primitive:

    Frame_t  (B, 3, H, W)
       │
       ▼   VisionEncoder (patchify + QCompassBi blocks)          — existing
    patches (B, N, C)
       │
       ▼   StateEncoder (QCompassBi with query token)             — existing
    state_t  (B, C)
       │       ┌──  action_emb  (B, action_dim)
       ▼       ▼
    TransitionModel (QCompassBi blocks)                           — existing
    state_{t+1}  (B, C)
       │
       ▼   PatchDecoder (QCompassBi blocks + patch unembed)      — NEW
    patches_{t+1}  (B, N, C)  →  reshape / unfold to
    Frame_{t+1}  (B, 3, H, W)

Loss: MSE on pixel reconstruction of Frame_{t+1}.
Target image is provided at training time, not used at inference.

Parameter budget (default config, 128×128 images, 8×8 patches, C=384):
  VisionEncoder   : ~4.6M
  StateEncoder    : ~1.2M
  TransitionModel : ~3.5M  (4 Q-Compass blocks)
  PatchDecoder    : ~5.5M  (4 Q-Compass blocks + patch projector)
  Action embed    : ~0.1M
  Total           : ~14.9M  (matches the "15M" design target)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuatrixConfig
from .vision import QCompassBi, QuatrixVisionBlock


class PatchDecoder(nn.Module):
    """Generate an image from a single state vector.

    state  (B, C)
      → broadcast to N patch tokens + learned positional bias
      → L_dec Q-Compass (bidirectional) blocks
      → linear projection  C → (patch_size² · 3)
      → unfold to RGB image

    Args:
        hidden_size : width of the latent patch tokens
        n_patches   : number of image patches (e.g. 64 for 128×128 at 16×16)
        patch_size  : side length of each patch in pixels
        n_layers    : number of Q-Compass decoder blocks
        q_rank      : compass rank
    """

    def __init__(self, hidden_size: int, n_patches: int, patch_size: int,
                 n_layers: int = 4, q_rank: int = 48, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_patches   = n_patches
        self.patch_size  = patch_size
        side = int(math.isqrt(n_patches))
        assert side * side == n_patches, \
            f"n_patches {n_patches} must be a perfect square"
        self.side = side

        # Learnable query tokens (one per patch) — broadcast state into positions
        self.patch_queries = nn.Parameter(
            torch.randn(n_patches, hidden_size) * 0.02)

        # Project state vector to "seed" each patch with global info
        self.state_broadcast = nn.Linear(hidden_size, hidden_size, bias=False)

        # Bidirectional Q-Compass decoder blocks
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(hidden_size, q_rank, ffn_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        # Final projection: each latent patch → (patch_size^2 × 3) pixels
        self.patch_head = nn.Linear(hidden_size, patch_size * patch_size * 3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, C)
        Returns:
            image: (B, 3, H, W)   where  H = W = side × patch_size
        """
        B = state.size(0)
        # Seed: broadcast state to each patch position + add learned queries
        seed = self.state_broadcast(state).unsqueeze(1)           # (B, 1, C)
        tokens = self.patch_queries.unsqueeze(0) + seed           # (B, N, C)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)                                 # (B, N, C)

        # (B, N, patch² × 3)
        pixels = self.patch_head(tokens)
        # Unfold to image: (B, side, side, patch, patch, 3) → (B, 3, H, W)
        pixels = pixels.view(B, self.side, self.side,
                             self.patch_size, self.patch_size, 3)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4).contiguous()     # (B, 3, side, patch, side, patch)
        img = pixels.view(B, 3,
                          self.side * self.patch_size,
                          self.side * self.patch_size)
        return img


# ── Minimal in-module vision encoder at custom resolution ───────────────────
# (The project's main VisionEncoder assumes 224×224; this version is
#  parameterised for the demo's 128×128 frames.)

class _SmallVisionEncoder(nn.Module):
    """Patch-based vision encoder parameterised for arbitrary square images."""

    def __init__(self, hidden_size: int, image_size: int = 128,
                 patch_size: int = 16, n_layers: int = 3,
                 q_rank: int = 48, dropout: float = 0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size   = image_size
        self.patch_size   = patch_size
        self.n_patches    = (image_size // patch_size) ** 2
        self.hidden_size  = hidden_size

        self.patch_embed = nn.Conv2d(3, hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size, bias=False)
        self.pos_emb = nn.Embedding(self.n_patches, hidden_size)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            QuatrixVisionBlock(hidden_size, q_rank, ffn_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img)                     # (B, C, side, side)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)              # (B, N, C)
        pos = torch.arange(self.n_patches, device=x.device).unsqueeze(0)
        x = self.drop(x + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _StateEncoder(nn.Module):
    """Compresses patch sequence (B, N, C) → state vector (B, C) via query token."""
    def __init__(self, hidden_size: int, q_rank: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.compass = QCompassBi(hidden_size, q_rank, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.size(0)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, patches], dim=1)
        x = self.norm(x + self.compass(x))
        return x[:, 0, :]


class _TransitionModel(nn.Module):
    """state × action → next state."""
    def __init__(self, hidden_size: int, action_dim: int,
                 n_layers: int, q_rank: int, dropout: float = 0.1):
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

    def forward(self, state: torch.Tensor,
                action_emb: torch.Tensor) -> torch.Tensor:
        s = self.state_proj(state)
        a = self.action_proj(action_emb)
        x = self.fusion(torch.cat([s, a], dim=-1)).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).squeeze(1)
        return self.out(x)


# ── Full model ──────────────────────────────────────────────────────────────
class QuatrixWorldGenerative(nn.Module):
    """Full generative world model built entirely on Q-Compass primitives."""

    def __init__(self,
                 image_size: int = 128,
                 patch_size: int = 16,
                 hidden_size: int = 384,
                 q_rank: int = 48,
                 encoder_layers: int = 3,
                 transition_layers: int = 4,
                 decoder_layers: int = 4,
                 action_dim: int = 64,
                 num_actions: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = _SmallVisionEncoder(
            hidden_size, image_size, patch_size,
            n_layers=encoder_layers, q_rank=q_rank, dropout=dropout)
        self.state_enc = _StateEncoder(hidden_size, q_rank, dropout)
        self.action_emb = nn.Embedding(num_actions, action_dim)
        self.transition = _TransitionModel(
            hidden_size, action_dim, transition_layers, q_rank, dropout)
        self.decoder = PatchDecoder(
            hidden_size, self.encoder.n_patches, patch_size,
            n_layers=decoder_layers, q_rank=q_rank, dropout=dropout)

        self.image_size = image_size
        self.patch_size = patch_size
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        patches = self.encoder(frame)
        return self.state_enc(patches)

    def forward(self, frame_t: torch.Tensor, action_ids: torch.Tensor,
                frame_t1: Optional[torch.Tensor] = None):
        """
        Args:
            frame_t   : (B, 3, H, W)
            action_ids: (B,) long
            frame_t1  : (B, 3, H, W) target frame, optional (training only)
        Returns:
            dict with 'pred_frame', 'pred_state', and 'loss' if frame_t1 given.
        """
        s_t  = self.encode(frame_t)
        a    = self.action_emb(action_ids)
        s_t1 = self.transition(s_t, a)
        pred = self.decoder(s_t1)

        loss = None
        if frame_t1 is not None:
            loss = F.mse_loss(pred, frame_t1)

        return {
            "pred_frame": pred,
            "pred_state": s_t1,
            "loss":       loss,
        }

    @torch.no_grad()
    def rollout(self, frame_0: torch.Tensor, action_seq: torch.Tensor):
        """Auto-regressive rollout: each predicted frame becomes the next input.

        Args:
            frame_0   : (1, 3, H, W) starting frame
            action_seq: (T,) long sequence of actions
        Returns:
            (T+1, 3, H, W) sequence of frames (index 0 = input)
        """
        self.eval()
        frames = [frame_0.squeeze(0)]
        cur = frame_0
        for t in range(action_seq.size(0)):
            a = action_seq[t:t+1]
            out = self.forward(cur, a, frame_t1=None)
            cur = out["pred_frame"].clamp(0, 1)
            frames.append(cur.squeeze(0))
        return torch.stack(frames, dim=0)
