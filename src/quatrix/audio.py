"""QuatrixAudio — audio encoder built from the same QCompass blocks as QuatrixLM.

No Whisper, no external dependencies. Fully self-contained.

Audio → mel-spectrogram → time-frequency patches → QCompassBi blocks → audio tokens
Audio tokens are prepended to text tokens inside QuatrixLM (same as vision).

Input: raw waveform or pre-computed mel-spectrogram
Output: [B, N_patches, lm_hidden] — ready to prepend to text tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import QCompassBi, QuatrixVisionBlock  # same blocks, reused


# ── Config ────────────────────────────────────────────────────────────────────
AUD_HIDDEN   = 384     # audio encoder hidden size
AUD_LAYERS   = 3       # number of QCompassBi blocks
AUD_Q_RANK   = 32      # navigation rank
N_MELS       = 80      # mel filterbank bins (standard: 80)
PATCH_TIME   = 16      # time frames per patch
PATCH_FREQ   = 16      # frequency bins per patch
MAX_MEL_LEN  = 3000    # max mel frames (~30s at 100fps)
# ─────────────────────────────────────────────────────────────────────────────


class MelPatchEmbed(nn.Module):
    """
    Mel-spectrogram → 2D patches → patch embeddings.

    Treats the mel-spectrogram as a 2D image:
      [B, 1, N_MELS, T] → patches → [B, N_patches, AUD_HIDDEN]

    Same idea as vision patch embedding but on spectrogram.
    """

    def __init__(self, n_mels: int = N_MELS, patch_freq: int = PATCH_FREQ,
                 patch_time: int = PATCH_TIME, hidden_size: int = AUD_HIDDEN):
        super().__init__()
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        # Conv2d on (freq, time) dimensions
        self.proj = nn.Conv2d(
            in_channels  = 1,
            out_channels = hidden_size,
            kernel_size  = (patch_freq, patch_time),
            stride       = (patch_freq, patch_time),
            bias         = False,
        )
        self.n_freq_patches = n_mels // patch_freq   # 80 // 16 = 5

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, N_MELS, T] — mel-spectrogram
        Returns:
            [B, N_patches, AUD_HIDDEN]
        """
        x = self.proj(mel)                           # [B, AUD_HIDDEN, n_freq, n_time]
        B, C, nf, nt = x.shape
        x = x.permute(0, 2, 3, 1)                   # [B, n_freq, n_time, C]
        x = x.reshape(B, nf * nt, C)                # [B, N_patches, AUD_HIDDEN]
        return x


class AudioEncoder(nn.Module):
    """
    QuatrixAudio — mel-spectrogram encoder using Q-Compass blocks.

    Audio waveform → mel-spectrogram → patches → QCompassBi blocks → [B, N, lm_hidden]

    No Whisper, no external model. Trained end-to-end with QuatrixLM.
    Compatible with QuatrixVision — same token prepending mechanism.

    Usage in QuatrixLM:
        audio_tokens = audio_encoder(mel_spectrogram)   # [B, N, H]
        x = cat([audio_tokens, vision_tokens, text_tokens], dim=1)
    """

    def __init__(self, lm_hidden: int = 512, n_mels: int = N_MELS,
                 dropout: float = 0.1):
        super().__init__()
        self.lm_hidden = lm_hidden

        # Patch embedding
        self.patch_embed = MelPatchEmbed(
            n_mels=n_mels,
            patch_freq=PATCH_FREQ,
            patch_time=PATCH_TIME,
            hidden_size=AUD_HIDDEN,
        )

        # Learnable positional embeddings (max patches from 30s audio)
        n_freq_patches  = n_mels // PATCH_FREQ           # 5
        n_time_patches  = MAX_MEL_LEN // PATCH_TIME       # 187
        self.max_patches = n_freq_patches * n_time_patches
        self.pos_emb = nn.Embedding(self.max_patches, AUD_HIDDEN)

        self.drop = nn.Dropout(dropout)

        # Q-Compass audio blocks (bidirectional — audio has no causal constraint)
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(AUD_HIDDEN, AUD_Q_RANK, ffn_ratio=4, dropout=dropout)
            for _ in range(AUD_LAYERS)
        ])

        self.norm = nn.LayerNorm(AUD_HIDDEN)

        # Project from audio hidden to LM hidden
        self.proj = nn.Linear(AUD_HIDDEN, lm_hidden, bias=False)

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

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, N_MELS, T] — mel-spectrogram
                 T can vary (different audio lengths)
        Returns:
            [B, N_patches, lm_hidden] — audio tokens ready to prepend to text
        """
        x = self.patch_embed(mel)                    # [B, N_patches, AUD_HIDDEN]
        B, N, _ = x.shape

        if N > self.max_patches:
            raise ValueError(
                f"Audio produces {N} patches but AudioEncoder supports at most "
                f"{self.max_patches} (~30s). Truncate input audio before encoding."
            )

        pos = torch.arange(N, device=x.device).unsqueeze(0)
        x = self.drop(x + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.proj(x)                          # [B, N_patches, lm_hidden]


def waveform_to_mel(waveform: torch.Tensor, sample_rate: int = 16000,
                    n_mels: int = N_MELS, n_fft: int = 400,
                    hop_length: int = 160) -> torch.Tensor:
    """
    Convert raw waveform to mel-spectrogram.
    Requires torchaudio.

    Args:
        waveform: [B, T] or [T] — raw audio samples
        sample_rate: audio sample rate (default 16kHz)
    Returns:
        [B, 1, N_MELS, T_frames]
    """
    try:
        import torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(waveform.device)
        mel = mel_transform(waveform)                # [B, N_MELS, T]
        mel = (mel + 1e-6).log()                    # log-mel
        return mel.unsqueeze(1)                      # [B, 1, N_MELS, T]
    except ImportError:
        raise ImportError("torchaudio required for waveform_to_mel. "
                          "Install with: pip install torchaudio")
