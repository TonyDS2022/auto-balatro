"""Evaluation utilities for Balatro RL agent.

Provides tools for evaluating trained policies, computing metrics,
and comparing different models or strategies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.agent.policy import ActorCritic
from src.game.environment import BalatroEnv

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode.

    Attributes:
        total_reward: Sum of rewards in episode.
        length: Number of steps in episode.
        final_ante: Ante reached at episode end.
        final_score: Score at episode end.
        money_earned: Total money earned.
        hands_played: Total hands played.
        won: Whether the episode was won (completed all antes).
    """

    total_reward: float
    length: int
    final_ante: int
    final_score: int
    money_earned: int
    hands_played: int
    won: bool


@dataclass
class EvaluationResult:
    """Aggregated results from multiple evaluation episodes.

    Attributes:
        num_episodes: Number of episodes evaluated.
        mean_reward: Mean episode reward.
        std_reward: Standard deviation of rewards.
        mean_length: Mean episode length.
        mean_ante: Mean final ante reached.
        max_ante: Maximum ante reached.
        win_rate: Fraction of episodes won.
        episodes: Individual episode results.
        elapsed_time: Total evaluation time in seconds.
    """

    num_episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    mean_ante: float
    max_ante: int
    win_rate: float
    episodes: list[EpisodeResult] = field(default_factory=list)
    elapsed_time: float = 0.0

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Evaluation ({self.num_episodes} episodes):\n"
            f"  Reward: {self.mean_reward:.1f} ± {self.std_reward:.1f}\n"
            f"  Length: {self.mean_length:.0f} steps\n"
            f"  Ante: {self.mean_ante:.1f} (max: {self.max_ante})\n"
            f"  Win Rate: {self.win_rate * 100:.1f}%"
        )


class PolicyEvaluator:
    """Evaluates trained policies on Balatro.

    Runs multiple episodes and computes statistics.
    """

    def __init__(
        self,
        max_ante: int = 8,
        device: str = "auto",
    ) -> None:
        """Initialize evaluator.

        Args:
            max_ante: Maximum ante for evaluation.
            device: Device to run policy on.
        """
        self.max_ante = max_ante

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"PolicyEvaluator initialized: max_ante={max_ante}, device={self.device}")

    def evaluate(
        self,
        policy: ActorCritic,
        num_episodes: int = 100,
        deterministic: bool = True,
        seed: int | None = None,
        render: bool = False,
    ) -> EvaluationResult:
        """Evaluate a policy over multiple episodes.

        Args:
            policy: Policy to evaluate.
            num_episodes: Number of episodes to run.
            deterministic: Use deterministic actions (argmax).
            seed: Random seed for reproducibility.
            render: Whether to render episodes.

        Returns:
            EvaluationResult with aggregated statistics.
        """
        policy.eval()
        policy.to(self.device)

        env = BalatroEnv(
            max_ante=self.max_ante,
            render_mode="human" if render else None,
            seed=seed,
        )

        episodes: list[EpisodeResult] = []
        start_time = time.time()

        for ep in range(num_episodes):
            result = self._run_episode(policy, env, deterministic)
            episodes.append(result)

            if (ep + 1) % 10 == 0:
                logger.debug(f"Evaluated {ep + 1}/{num_episodes} episodes")

        elapsed = time.time() - start_time

        # Aggregate statistics
        rewards = [e.total_reward for e in episodes]
        lengths = [e.length for e in episodes]
        antes = [e.final_ante for e in episodes]
        wins = [e.won for e in episodes]

        return EvaluationResult(
            num_episodes=num_episodes,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_length=float(np.mean(lengths)),
            mean_ante=float(np.mean(antes)),
            max_ante=max(antes),
            win_rate=sum(wins) / len(wins),
            episodes=episodes,
            elapsed_time=elapsed,
        )

    def _run_episode(
        self,
        policy: ActorCritic,
        env: BalatroEnv,
        deterministic: bool,
    ) -> EpisodeResult:
        """Run a single evaluation episode.

        Args:
            policy: Policy to use.
            env: Environment instance.
            deterministic: Use deterministic actions.

        Returns:
            EpisodeResult for the episode.
        """
        obs, info = env.reset()
        total_reward = 0.0
        length = 0
        hands_played = 0

        while True:
            # Get action mask
            action_mask = info.get("valid_actions", np.ones(15, dtype=bool))

            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                action, _, _ = policy.get_action(
                    obs_tensor, mask_tensor, deterministic=deterministic
                )
                action = action.item()

            # Track hand plays
            if action == 0:  # Play hand action
                hands_played += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1

            if terminated or truncated:
                break

        # Determine if won (reached max ante)
        final_ante = info.get("ante", 1)
        won = final_ante > self.max_ante or (terminated and final_ante == self.max_ante)

        return EpisodeResult(
            total_reward=total_reward,
            length=length,
            final_ante=final_ante,
            final_score=info.get("score", 0),
            money_earned=info.get("money", 0),
            hands_played=hands_played,
            won=won,
        )

    def compare_policies(
        self,
        policies: dict[str, ActorCritic],
        num_episodes: int = 50,
        seed: int | None = 42,
    ) -> dict[str, EvaluationResult]:
        """Compare multiple policies.

        Args:
            policies: Dict mapping policy names to policies.
            num_episodes: Episodes per policy.
            seed: Seed for fair comparison.

        Returns:
            Dict mapping policy names to results.
        """
        results = {}

        for name, policy in policies.items():
            logger.info(f"Evaluating policy: {name}")
            results[name] = self.evaluate(
                policy,
                num_episodes=num_episodes,
                seed=seed,
            )
            logger.info(f"{name}: {results[name].mean_reward:.1f} reward, {results[name].win_rate*100:.1f}% win")

        return results


class RandomBaseline:
    """Random action baseline for comparison."""

    def __init__(self, action_dim: int = 15) -> None:
        """Initialize random baseline."""
        self.action_dim = action_dim

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get random valid action.

        Args:
            obs: Observation (ignored).
            action_mask: Valid action mask.
            deterministic: Ignored for random baseline.

        Returns:
            Tuple of (action, log_prob, value).
        """
        batch_size = obs.shape[0]

        if action_mask is not None:
            # Sample from valid actions only
            actions = []
            for i in range(batch_size):
                valid_indices = torch.where(action_mask[i])[0]
                if len(valid_indices) > 0:
                    idx = torch.randint(len(valid_indices), (1,)).item()
                    actions.append(valid_indices[idx].item())
                else:
                    actions.append(0)
            action = torch.tensor(actions)
        else:
            action = torch.randint(0, self.action_dim, (batch_size,))

        # Dummy values
        log_prob = torch.zeros(batch_size)
        value = torch.zeros(batch_size)

        return action, log_prob, value

    def eval(self) -> "RandomBaseline":
        """No-op for compatibility."""
        return self

    def to(self, device) -> "RandomBaseline":
        """No-op for compatibility."""
        return self


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    num_episodes: int = 100,
    max_ante: int = 8,
    device: str = "auto",
) -> EvaluationResult:
    """Evaluate a saved checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        num_episodes: Number of evaluation episodes.
        max_ante: Maximum ante for evaluation.
        device: Device to run on.

    Returns:
        EvaluationResult.
    """
    # Load checkpoint
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create policy and load weights
    policy = ActorCritic()
    policy.load_state_dict(checkpoint["policy_state_dict"])

    # Evaluate
    evaluator = PolicyEvaluator(max_ante=max_ante, device=device)
    return evaluator.evaluate(policy, num_episodes=num_episodes)


def benchmark_against_random(
    policy: ActorCritic,
    num_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Benchmark a policy against random baseline.

    Args:
        policy: Policy to benchmark.
        num_episodes: Episodes to run.
        seed: Random seed.

    Returns:
        Dict with comparison statistics.
    """
    evaluator = PolicyEvaluator()

    # Evaluate policy
    policy_result = evaluator.evaluate(policy, num_episodes=num_episodes, seed=seed)

    # Evaluate random baseline
    random_policy = RandomBaseline()
    random_result = evaluator.evaluate(random_policy, num_episodes=num_episodes, seed=seed)

    # Compute improvement
    reward_improvement = policy_result.mean_reward - random_result.mean_reward
    ante_improvement = policy_result.mean_ante - random_result.mean_ante
    win_improvement = policy_result.win_rate - random_result.win_rate

    return {
        "policy_reward": policy_result.mean_reward,
        "random_reward": random_result.mean_reward,
        "reward_improvement": reward_improvement,
        "policy_ante": policy_result.mean_ante,
        "random_ante": random_result.mean_ante,
        "ante_improvement": ante_improvement,
        "policy_win_rate": policy_result.win_rate,
        "random_win_rate": random_result.win_rate,
        "win_improvement": win_improvement,
    }
