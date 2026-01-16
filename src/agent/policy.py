"""Neural network policies for Balatro RL agent.

Implements actor-critic networks for PPO training with support
for action masking (invalid action filtering).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BalatroNetwork(nn.Module):
    """Shared feature extractor for actor-critic.

    Processes game observations into feature representations
    used by both policy (actor) and value (critic) heads.

    Architecture:
    - Input: Observation vector (69 dims from BalatroEnv)
    - Hidden: 2 fully connected layers with ReLU
    - Output: Feature vector for actor/critic heads
    """

    def __init__(
        self,
        obs_dim: int = 69,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the network.

        Args:
            obs_dim: Observation space dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization for training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in [self.fc1, self.fc2]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Feature tensor of shape (batch, hidden_dim).
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return x


class ActorHead(nn.Module):
    """Policy head that outputs action probabilities.

    Supports action masking to prevent selection of invalid actions.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 15,
    ) -> None:
        """Initialize actor head.

        Args:
            feature_dim: Input feature dimension.
            action_dim: Number of discrete actions.
        """
        super().__init__()

        self.fc = nn.Linear(feature_dim, action_dim)
        self.action_dim = action_dim

        nn.init.orthogonal_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action logits with optional masking.

        Args:
            features: Feature tensor from backbone.
            action_mask: Boolean mask where True = valid action.

        Returns:
            Action logits (masked if mask provided).
        """
        logits = self.fc(features)

        if action_mask is not None:
            # Set invalid action logits to large negative value
            invalid_mask = ~action_mask
            logits = logits.masked_fill(invalid_mask, -1e8)

        return logits


class CriticHead(nn.Module):
    """Value head that estimates state value."""

    def __init__(self, feature_dim: int = 256) -> None:
        """Initialize critic head.

        Args:
            feature_dim: Input feature dimension.
        """
        super().__init__()

        self.fc = nn.Linear(feature_dim, 1)

        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            features: Feature tensor from backbone.

        Returns:
            Value tensor of shape (batch, 1).
        """
        return self.fc(features)


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO.

    Architecture:
    - Shared feature backbone (BalatroNetwork)
    - Actor head for policy (action probabilities)
    - Critic head for value estimation

    Supports action masking for valid action filtering.
    """

    def __init__(
        self,
        obs_dim: int = 69,
        action_dim: int = 15,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize actor-critic network.

        Args:
            obs_dim: Observation space dimension.
            action_dim: Number of discrete actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.backbone = BalatroNetwork(obs_dim, hidden_dim)
        self.actor = ActorHead(hidden_dim, action_dim)
        self.critic = CriticHead(hidden_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        logger.info(
            f"ActorCritic initialized: obs={obs_dim}, actions={action_dim}, hidden={hidden_dim}"
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value.

        Args:
            obs: Observation tensor.
            action_mask: Optional valid action mask.

        Returns:
            Tuple of (action_logits, state_value).
        """
        features = self.backbone(obs)
        logits = self.actor(features, action_mask)
        value = self.critic(features)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            obs: Observation tensor.
            action_mask: Optional valid action mask.
            deterministic: If True, return argmax action.

        Returns:
            Tuple of (action, log_prob, value).
        """
        logits, value = self.forward(obs, action_mask)

        if deterministic:
            action = logits.argmax(dim=-1)
            # Compute log prob for the selected action
            probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            obs: Observation tensor.
            actions: Action tensor.
            action_mask: Optional valid action mask.

        Returns:
            Tuple of (log_probs, values, entropy).
        """
        logits, value = self.forward(obs, action_mask)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value.squeeze(-1), entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value only (for bootstrapping).

        Args:
            obs: Observation tensor.

        Returns:
            Value tensor.
        """
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)


class CardSelectionNetwork(nn.Module):
    """Specialized network for card selection decisions.

    Uses attention mechanism to reason about card combinations.
    This is an optional enhancement over the basic ActorCritic.
    """

    def __init__(
        self,
        card_features: int = 6,
        max_cards: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        """Initialize card selection network.

        Args:
            card_features: Features per card (rank, suit one-hot, enhancement).
            max_cards: Maximum cards in hand.
            hidden_dim: Hidden dimension.
            num_heads: Attention heads.
        """
        super().__init__()

        self.max_cards = max_cards
        self.card_features = card_features

        # Card embedding
        self.card_embed = nn.Linear(card_features, hidden_dim)

        # Self-attention over cards
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output: score for each card
        self.card_score = nn.Linear(hidden_dim, 1)

        logger.info(f"CardSelectionNetwork initialized: {max_cards} cards, {hidden_dim} hidden")

    def forward(
        self,
        card_obs: torch.Tensor,
        card_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute card selection scores.

        Args:
            card_obs: Card observations of shape (batch, max_cards, card_features).
            card_mask: Mask for valid cards (batch, max_cards).

        Returns:
            Selection scores of shape (batch, max_cards).
        """
        # Embed cards
        embedded = F.relu(self.card_embed(card_obs))

        # Self-attention
        # Create attention mask (True = ignore)
        if card_mask is not None:
            attn_mask = ~card_mask
        else:
            attn_mask = None

        attended, _ = self.attention(
            embedded, embedded, embedded,
            key_padding_mask=attn_mask,
        )

        # Score each card
        scores = self.card_score(attended).squeeze(-1)

        # Mask invalid cards
        if card_mask is not None:
            scores = scores.masked_fill(~card_mask, -1e8)

        return scores


def create_policy(
    obs_dim: int = 69,
    action_dim: int = 15,
    hidden_dim: int = 256,
    device: str = "auto",
) -> ActorCritic:
    """Create and initialize an ActorCritic policy.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer size.
        device: Device to place model on ("auto", "cpu", "cuda").

    Returns:
        Initialized ActorCritic model.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(obs_dim, action_dim, hidden_dim)
    model = model.to(device)

    logger.info(f"Created policy on device: {device}")
    return model
