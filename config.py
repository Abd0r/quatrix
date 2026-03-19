"""QuatrixConfig — Model configuration."""

from dataclasses import dataclass


@dataclass
class QuatrixConfig:
    """
    Configuration for QuatrixLM.

    Q-Compass architecture: state × action navigation matrix,
    no value projection, gathers directly from x.
    """
    vocab_size: int = 50257       # GPT-2 default
    hidden_size: int = 512
    num_layers: int = 7           # 7 LM layers + 3 vision layers ≈ 50M total
    max_seq_len: int = 5120       # 5K context (5120 = 5 × 1024)
    q_rank: int = 64              # compass projection rank
    ffn_ratio: int = 4            # FFN hidden = hidden_size * ffn_ratio
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    # Modality plugins — all self-contained, no external dependencies
    use_vision:      bool = False   # QuatrixVision — image patches
    use_audio:       bool = False   # QuatrixAudio  — mel-spectrogram patches
    use_world_model: bool = False   # QuatrixWorld  — state-action-transition plugin
