"""QuatrixWorld — World Model plugin for QuatrixLM.

Extends QuatrixLM with temporal state modeling and action prediction.
Enables Berry-Q1+ to reason about: "if I do X, what happens next?"

Architecture:
  State encoder   : encodes current world state (text + vision + audio tokens)
  Action head     : predicts next action given state
  Transition model: predicts next state given current state + action  Q(s,a)
  Reward head     : estimates value/reward of a state (optional, for RL)

The World Model is a PLUGIN — it wraps QuatrixLM without modifying it.
QuatrixLM stays frozen or trains jointly depending on use case.

Usage:
    from quatrix import QuatrixLM, QuatrixConfig
    from quatrix.world import WorldModel

    lm = QuatrixLM(cfg)
    world = WorldModel(lm, action_dim=256, state_dim=512)

    # Given current state tokens, predict next state + action
    next_state, action_logits = world(state_tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import QCompassBi, QuatrixVisionBlock


# ── Config ────────────────────────────────────────────────────────────────────
WM_HIDDEN      = 512    # world model hidden (matches LM hidden)
WM_LAYERS      = 4      # transition model depth
WM_Q_RANK      = 64     # navigation rank for world model
WM_ACTION_DIM  = 256    # action embedding dimension
WM_STATE_DIM   = 512    # state embedding dimension
# ─────────────────────────────────────────────────────────────────────────────


class StateEncoder(nn.Module):
    """
    Compress a sequence of tokens into a single state vector.
    Uses Q-Compass to aggregate the most valuable information.

    Input:  [B, L, H] — token sequence from QuatrixLM
    Output: [B, H]    — compressed state vector
    """

    def __init__(self, hidden_size: int = WM_HIDDEN, q_rank: int = WM_Q_RANK,
                 dropout: float = 0.1):
        super().__init__()
        self.compass = QCompassBi(hidden_size, q_rank, dropout)
        self.norm    = nn.LayerNorm(hidden_size)
        # Learnable "query" token that aggregates the sequence
        self.query   = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, H] — token sequence
        Returns:
            [B, H] — state vector
        """
        B = x.size(0)
        # Prepend learnable query token
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, x], dim=1)              # [B, L+1, H]
        x = self.norm(x + self.compass(x))
        return x[:, 0, :]                          # [B, H] — query token output


class TransitionModel(nn.Module):
    """
    Predicts the next world state given current state + action.

    Implements: s' = f(s, a)  — the core world model function.

    Uses Q-Compass blocks — the Q(s,a) navigation naturally models
    state-action transitions (same math as RL Q-function).

    Input:  state [B, H], action [B, A]
    Output: next_state [B, H]
    """

    def __init__(self, state_dim: int = WM_STATE_DIM,
                 action_dim: int = WM_ACTION_DIM,
                 hidden_size: int = WM_HIDDEN,
                 n_layers: int = WM_LAYERS,
                 q_rank: int = WM_Q_RANK,
                 dropout: float = 0.1):
        super().__init__()

        # Project state + action into joint hidden space
        self.state_proj  = nn.Linear(state_dim, hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.fusion      = nn.Linear(hidden_size * 2, hidden_size)

        # Q-Compass transition blocks
        self.blocks = nn.ModuleList([
            QuatrixVisionBlock(hidden_size, q_rank, ffn_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.out  = nn.Linear(hidden_size, state_dim)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  [B, state_dim]
            action: [B, action_dim]
        Returns:
            next_state: [B, state_dim]
        """
        s = self.state_proj(state)                  # [B, H]
        a = self.action_proj(action)                # [B, H]
        x = self.fusion(torch.cat([s, a], dim=-1)) # [B, H]
        x = x.unsqueeze(1)                          # [B, 1, H] — treat as sequence

        for block in self.blocks:
            x = block(x)

        x = self.norm(x).squeeze(1)                 # [B, H]
        return self.out(x)                          # [B, state_dim]


class ActionHead(nn.Module):
    """
    Predicts action distribution given current state.

    For discrete actions: outputs logits over action vocabulary.
    For continuous actions: outputs mean + log_std of Gaussian.

    Input:  [B, H] — state vector
    Output: [B, action_dim] — action logits or parameters
    """

    def __init__(self, state_dim: int = WM_STATE_DIM,
                 action_dim: int = WM_ACTION_DIM,
                 continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, action_dim * (2 if continuous else 1)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            [B, action_dim] — logits (discrete) or [mean, log_std] (continuous)
        """
        return self.net(state)


class RewardHead(nn.Module):
    """
    Estimates scalar reward/value for a given state.
    Optional — used for RL fine-tuning.

    Input:  [B, H] — state vector
    Output: [B, 1] — scalar value estimate
    """

    def __init__(self, state_dim: int = WM_STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)                      # [B, 1]


class WorldModel(nn.Module):
    """
    QuatrixWorld — Full World Model plugin wrapping QuatrixLM.

    Adds temporal reasoning to Berry-Q1+:
      - Encode current state from LM hidden states
      - Predict next action
      - Predict next world state after action
      - Optionally estimate reward/value

    The Q(s,a) math in Q-Compass IS a world model Q-function.
    This module makes that explicit and trainable on world model data.

    All dimensions (hidden, q_rank, state_dim) default to `lm_hidden`-derived
    values so the world model scales 1:1 with the language model.

    Usage:
        world = WorldModel(lm_hidden=512, n_transition_layers=10, q_rank=64)
        state_vec, action_logits, next_state, reward = world(hidden_states, action)
    """

    def __init__(self, lm_hidden: int = 512,
                 action_dim: int = WM_ACTION_DIM,
                 n_transition_layers: int = WM_LAYERS,
                 q_rank: int = WM_Q_RANK,
                 use_reward_head: bool = False,
                 continuous_actions: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        # State dimension is tied to lm_hidden so world-model state vectors
        # flow naturally through QuatrixLM hidden states
        state_dim = lm_hidden

        self.state_encoder  = StateEncoder(lm_hidden, q_rank, dropout)
        self.transition     = TransitionModel(
            state_dim, action_dim, lm_hidden, n_transition_layers, q_rank, dropout,
        )
        self.action_head    = ActionHead(state_dim, action_dim, continuous_actions)
        self.reward_head    = RewardHead(state_dim) if use_reward_head else None

    def forward(self, hidden_states: torch.Tensor,
                action: torch.Tensor = None):
        """
        Args:
            hidden_states: [B, L, H] — LM hidden states for current context
            action: [B, action_dim] — taken action (for transition prediction)
                    If None, only predicts action, not next state.
        Returns:
            state_vec:      [B, state_dim]   — current world state
            action_logits:  [B, action_dim]  — predicted action distribution
            next_state:     [B, state_dim]   — predicted next state (if action given)
            reward:         [B, 1]           — value estimate (if reward_head enabled)
        """
        # Encode current state
        state_vec     = self.state_encoder(hidden_states)   # [B, H]
        action_logits = self.action_head(state_vec)         # [B, action_dim]

        next_state = None
        if action is not None:
            next_state = self.transition(state_vec, action) # [B, state_dim]

        reward = None
        if self.reward_head is not None:
            reward = self.reward_head(state_vec)            # [B, 1]

        return state_vec, action_logits, next_state, reward
