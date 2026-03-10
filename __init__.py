"""Quatrix — Q-Compass Architecture.

Modalities:
  - Text  : QuatrixLM  (causal language model)
  - Vision: VisionEncoder (image patches → tokens)
  - Audio : AudioEncoder  (mel-spectrogram → tokens)
  - World : WorldModel    (state-action-transition plugin)
"""

from .config import QuatrixConfig
from .model  import QCompass, QuatrixBlock, QuatrixLM
from .vision import QCompassBi, QuatrixVisionBlock, VisionEncoder
from .audio  import AudioEncoder, waveform_to_mel
from .world  import WorldModel, StateEncoder, TransitionModel, ActionHead, RewardHead

__all__ = [
    # Config
    "QuatrixConfig",
    # LM
    "QCompass",
    "QuatrixBlock",
    "QuatrixLM",
    # Vision
    "QCompassBi",
    "QuatrixVisionBlock",
    "VisionEncoder",
    # Audio
    "AudioEncoder",
    "waveform_to_mel",
    # World Model
    "WorldModel",
    "StateEncoder",
    "TransitionModel",
    "ActionHead",
    "RewardHead",
]
