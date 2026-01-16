"""PPO trainer for Balatro RL agent.

Implements Proximal Policy Optimization with:
- GAE (Generalized Advantage Estimation)
- Action masking for valid actions
- Curriculum learning (progressive ante difficulty)
- Checkpointing and logging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.policy import ActorCritic, create_policy
from src.game.environment import BalatroEnv

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for PPO trainer.

    Attributes:
        total_timesteps: Total environment steps to train.
        learning_rate: Optimizer learning rate.
        n_steps: Steps per rollout before update.
        batch_size: Minibatch size for updates.
        n_epochs: PPO epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_range: PPO clipping range.
        clip_range_vf: Value function clip range (None to disable).
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping norm.
        target_kl: Target KL divergence for early stopping.
        device: Device to train on.
    """

    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    device: str = "auto"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        enabled: Whether to use curriculum learning.
        initial_max_ante: Starting max ante.
        final_max_ante: Final max ante to reach.
        ante_threshold: Win rate needed to increase ante.
        evaluation_window: Episodes to evaluate win rate over.
    """

    enabled: bool = True
    initial_max_ante: int = 2
    final_max_ante: int = 8
    ante_threshold: float = 0.5
    evaluation_window: int = 100


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data.

    Stores observations, actions, rewards, etc. for PPO updates.
    """

    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    action_masks: list = field(default_factory=list)

    # Computed after rollout
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        action_mask: np.ndarray,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.action_masks.clear()
        self.advantages = None
        self.returns = None

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_batches(
        self,
        batch_size: int,
        device: torch.device,
    ):
        """Generate minibatches for training.

        Args:
            batch_size: Size of each minibatch.
            device: Device to place tensors on.

        Yields:
            Tuples of (obs, actions, old_log_probs, advantages, returns, action_masks).
        """
        n = len(self.observations)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                torch.FloatTensor(np.array([self.observations[i] for i in batch_indices])).to(device),
                torch.LongTensor(np.array([self.actions[i] for i in batch_indices])).to(device),
                torch.FloatTensor(np.array([self.log_probs[i] for i in batch_indices])).to(device),
                torch.FloatTensor(self.advantages[batch_indices]).to(device),
                torch.FloatTensor(self.returns[batch_indices]).to(device),
                torch.BoolTensor(np.array([self.action_masks[i] for i in batch_indices])).to(device),
            )


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Trains an actor-critic policy on Balatro using PPO with:
    - GAE advantage estimation
    - Action masking
    - Curriculum learning
    - Checkpoint saving
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        curriculum: CurriculumConfig | None = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration.
            curriculum: Curriculum learning configuration.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.config = config or TrainerConfig()
        self.curriculum = curriculum or CurriculumConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Initialize environment and policy
        self.current_max_ante = (
            self.curriculum.initial_max_ante
            if self.curriculum.enabled
            else 8
        )
        self.env = BalatroEnv(max_ante=self.current_max_ante)
        self.policy = create_policy(device=str(self.device))
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.total_timesteps = 0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_antes: list[int] = []

        logger.info(
            f"PPOTrainer initialized: device={self.device}, "
            f"curriculum={'enabled' if self.curriculum.enabled else 'disabled'}"
        )

    def collect_rollout(self) -> dict:
        """Collect a rollout of experience.

        Returns:
            Dict with rollout statistics.
        """
        self.buffer.clear()
        self.policy.eval()

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _ in range(self.config.n_steps):
            # Get action mask
            action_mask = info.get("valid_actions", np.ones(15, dtype=bool))

            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                action, log_prob, value = self.policy.get_action(
                    obs_tensor, mask_tensor
                )
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                action_mask=action_mask,
            )

            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1

            if done:
                # Record episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_antes.append(info.get("ante", 1))

                # Reset
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs

        # Compute last value for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.policy.get_value(obs_tensor).item()

        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return {
            "timesteps": self.config.n_steps,
            "episodes": len(self.episode_rewards) - len(self.episode_antes),
        }

    def train_step(self) -> dict:
        """Perform one PPO update.

        Returns:
            Dict with training statistics.
        """
        self.policy.train()

        # Track losses
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size, self.device):
                obs, actions, old_log_probs, advantages, returns, action_masks = batch

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy outputs
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    obs, actions, action_masks
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    old_values = returns - self.buffer.advantages[: len(returns)]
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss1 = (values - returns) ** 2
                    value_loss2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track stats
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

                # Compute KL divergence for early stopping
                with torch.no_grad():
                    kl = (old_log_probs - new_log_probs).mean().item()
                    kl_divs.append(kl)

            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                mean_kl = np.mean(kl_divs)
                if mean_kl > 1.5 * self.config.target_kl:
                    logger.debug(f"Early stopping at epoch {epoch} due to KL: {mean_kl:.4f}")
                    break

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "kl_divergence": np.mean(kl_divs),
        }

    def update_curriculum(self) -> bool:
        """Update curriculum based on recent performance.

        Returns:
            True if curriculum was updated.
        """
        if not self.curriculum.enabled:
            return False

        if self.current_max_ante >= self.curriculum.final_max_ante:
            return False

        # Check recent episodes
        window = self.curriculum.evaluation_window
        if len(self.episode_antes) < window:
            return False

        recent_antes = self.episode_antes[-window:]
        wins = sum(1 for ante in recent_antes if ante >= self.current_max_ante)
        win_rate = wins / window

        if win_rate >= self.curriculum.ante_threshold:
            self.current_max_ante += 1
            self.env = BalatroEnv(max_ante=self.current_max_ante)
            logger.info(f"Curriculum updated: max_ante={self.current_max_ante}")
            return True

        return False

    def train(self, callback=None) -> dict:
        """Run full training loop.

        Args:
            callback: Optional callback(trainer) called each iteration.

        Returns:
            Final training statistics.
        """
        start_time = time.time()
        iteration = 0

        logger.info(f"Starting training for {self.config.total_timesteps} timesteps")

        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout
            rollout_stats = self.collect_rollout()

            # Train on collected data
            train_stats = self.train_step()

            # Update curriculum
            self.update_curriculum()

            # Logging
            iteration += 1
            if iteration % 10 == 0:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0
                recent_antes = self.episode_antes[-100:]
                mean_ante = np.mean(recent_antes) if recent_antes else 0

                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed

                logger.info(
                    f"Iter {iteration} | Steps: {self.total_timesteps} | "
                    f"Reward: {mean_reward:.1f} | Ante: {mean_ante:.1f} | "
                    f"Policy Loss: {train_stats['policy_loss']:.4f} | "
                    f"FPS: {fps:.0f}"
                )

            # Checkpoint
            if iteration % 100 == 0:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")

            # Callback
            if callback is not None:
                callback(self)

        # Final checkpoint
        self.save_checkpoint("final.pt")

        elapsed = time.time() - start_time
        return {
            "total_timesteps": self.total_timesteps,
            "iterations": iteration,
            "elapsed_time": elapsed,
            "final_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "final_ante": np.mean(self.episode_antes[-100:]) if self.episode_antes else 0,
        }

    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
                "current_max_ante": self.current_max_ante,
                "config": self.config,
                "curriculum": self.curriculum,
            },
            path,
        )
        logger.debug(f"Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.current_max_ante = checkpoint["current_max_ante"]

        logger.info(f"Loaded checkpoint from {path}")
