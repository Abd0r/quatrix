"""QuatrixVision — image encoder built from the same QCompass blocks as QuatrixLM.

No CLIP, no external dependencies. Fully self-contained 50M model.

Image → patches → patch embedding → QuatrixBlocks (bidirectional) → patch tokens
Patch tokens are prepended to text tokens inside QuatrixLM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QCompassBi(nn.Module):
    """
    Bidirectional Q-Compass for image patches (no causal mask).
    Same mechanism as QCompass in the LM but attends in both directions.
    """

    def __init__(self, hidden_size: int, q_rank: int, dropout: float = 0.1):
        super().__init__()
        self.state_proj  = nn.Linear(hidden_size, q_rank, bias=False)
        self.action_proj = nn.Linear(hidden_size, q_rank, bias=False)
        self.out_proj    = nn.Linear(hidden_size, hidden_size)
        self.drop        = nn.Dropout(dropout)
        self.scale       = math.sqrt(q_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state  = self.state_proj(x)   # [B, N, q_rank]
        action = self.action_proj(x)  # [B, N, q_rank]
        q_vals = torch.bmm(state, action.transpose(-2, -1)) / self.scale  # [B, N, N]
        weights = F.softmax(q_vals, dim=-1)
        weights = self.drop(weights)
        out = torch.bmm(weights, x)   # [B, N, H] — gather directly from x
        return self.out_proj(out)


class QuatrixVisionBlock(nn.Module):
    """Pre-norm residual block for vision: QCompassBi + FFN."""

    def __init__(self, hidden_size: int, q_rank: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1   = nn.LayerNorm(hidden_size)
        self.compass = QCompassBi(hidden_size, q_rank, dropout)
        self.norm2   = nn.LayerNorm(hidden_size)
        ffn_dim = hidden_size * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.compass(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    QuatrixVision — patch-based image encoder using Q-Compass blocks.

    224×224 image → 196 patches (16×16) → QuatrixVisionBlocks → [B, 196, lm_hidden]

    No CLIP, no external vision model. Trained end-to-end with QuatrixLM.
    """

    PATCH_SIZE   = 16
    IMAGE_SIZE   = 224
    NUM_PATCHES  = (224 // 16) ** 2   # 196
    VIS_HIDDEN   = 384
    VIS_LAYERS   = 3
    VIS_Q_RANK   = 32

    def __init__(self, hidden_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Patch embedding: conv2d is equivalent to linear on each patch
        self.patch_embed = nn.Conv2d(
            in_channels  = 3,
            out_channels = self.VIS_HIDDEN,
            kernel_size  = self.PATCH_SIZE,
            stride       = self.PATCH_SIZE,
            bias         = False,
        )  # [B, VIS_HIDDEN, 14, 14] → reshape to [B, 196, VIS_HIDDEN]

        # Learnable positional embeddings for 196 patches
        self.pos_emb = nn.Embedding(self.NUM_PATCHES, self.VIS_HIDDEN)

        self.drop = nn.Dropout(dropout)

        # Q-Compass vision blocks (bidirectional)
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(self.VIS_HIDDEN, self.VIS_Q_RANK, ffn_ratio=4, dropout=dropout)
            for _ in range(self.VIS_LAYERS)
        ])

        self.norm = nn.LayerNorm(self.VIS_HIDDEN)

        # Project from vision hidden to LM hidden
        self.proj = nn.Linear(self.VIS_HIDDEN, hidden_size, bias=False)

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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 224, 224]
        Returns:
            [B, 196, hidden_size]  — patch tokens ready to prepend to text
        """
        B = pixel_values.size(0)

        # Patchify
        x = self.patch_embed(pixel_values)          # [B, VIS_HIDDEN, 14, 14]
        x = x.flatten(2).transpose(1, 2)            # [B, 196, VIS_HIDDEN]

        # Add positional embeddings
        pos = torch.arange(self.NUM_PATCHES, device=x.device).unsqueeze(0)
        x = self.drop(x + self.pos_emb(pos))

        # Q-Compass blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Project to LM hidden size
        return self.proj(x)                         # [B, 196, hidden_size]
