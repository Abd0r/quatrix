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
    use_vision:       bool = False  # QuatrixVision — image patches
    use_audio:        bool = False  # QuatrixAudio  — mel-spectrogram patches
    use_world_model:  bool = False  # QuatrixWorld  — state-action-transition plugin
    world_num_actions: int = 6     # discrete navigation actions (forward/back/left/right/look_up/look_down)
    wm_layers:        int = 0      # world-model transition layers (0 = same as num_layers)
    # wm_hidden and wm_q_rank implicitly follow hidden_size and q_rank

    # Q-Compass architectural extensions
    q_heads:           int  = 1     # multi-head Q-Compass (1 = single-head, default)
    q_value_content:   bool = False # D.1: gather self-Q-value (state⊙action·W_c) instead of raw x
