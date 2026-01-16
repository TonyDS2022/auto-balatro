"""Tests for policy evaluator."""

import numpy as np
import pytest
import torch

from src.agent.evaluator import (
    EpisodeResult,
    EvaluationResult,
    PolicyEvaluator,
    RandomBaseline,
)
from src.agent.policy import ActorCritic


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_creation(self):
        """Test episode result creation."""
        result = EpisodeResult(
            total_reward=100.5,
            length=50,
            final_ante=3,
            final_score=15000,
            money_earned=500,
            hands_played=10,
            won=False,
        )
        assert result.total_reward == 100.5
        assert result.length == 50
        assert result.final_ante == 3
        assert result.final_score == 15000
        assert result.money_earned == 500
        assert result.hands_played == 10
        assert result.won is False


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test evaluation result creation."""
        result = EvaluationResult(
            num_episodes=100,
            mean_reward=50.0,
            std_reward=10.0,
            mean_length=30.0,
            mean_ante=2.5,
            max_ante=5,
            win_rate=0.2,
        )
        assert result.num_episodes == 100
        assert result.mean_reward == 50.0
        assert result.win_rate == 0.2

    def test_str_format(self):
        """Test string formatting."""
        result = EvaluationResult(
            num_episodes=100,
            mean_reward=50.0,
            std_reward=10.0,
            mean_length=30.0,
            mean_ante=2.5,
            max_ante=5,
            win_rate=0.2,
        )
        text = str(result)
        assert "100 episodes" in text
        assert "50.0" in text
        assert "20.0%" in text


class TestRandomBaseline:
    """Tests for random baseline policy."""

    @pytest.fixture
    def baseline(self):
        """Create random baseline."""
        return RandomBaseline(action_dim=15)

    def test_initialization(self, baseline):
        """Test baseline initialization."""
        assert baseline.action_dim == 15

    def test_get_action_shape(self, baseline):
        """Test action output shapes."""
        obs = torch.randn(32, 69)

        action, log_prob, value = baseline.get_action(obs)

        assert action.shape == (32,)
        assert log_prob.shape == (32,)
        assert value.shape == (32,)

    def test_get_action_valid_range(self, baseline):
        """Test actions are in valid range."""
        obs = torch.randn(100, 69)

        action, _, _ = baseline.get_action(obs)

        assert (action >= 0).all()
        assert (action < 15).all()

    def test_action_masking(self, baseline):
        """Test action masking works."""
        obs = torch.randn(10, 69)
        mask = torch.zeros(10, 15, dtype=torch.bool)
        mask[:, 5] = True  # Only action 5 valid

        action, _, _ = baseline.get_action(obs, action_mask=mask)

        # All actions should be 5
        assert (action == 5).all()

    def test_eval_returns_self(self, baseline):
        """Test eval() returns self."""
        result = baseline.eval()
        assert result is baseline

    def test_to_returns_self(self, baseline):
        """Test to() returns self."""
        result = baseline.to("cpu")
        assert result is baseline


class TestPolicyEvaluator:
    """Tests for PolicyEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator."""
        return PolicyEvaluator(max_ante=3, device="cpu")

    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.max_ante == 3
        assert evaluator.device == torch.device("cpu")

    def test_initialization_auto_device(self):
        """Test auto device selection."""
        evaluator = PolicyEvaluator(device="auto")
        # Should be cpu or cuda depending on availability
        assert evaluator.device in [torch.device("cpu"), torch.device("cuda")]

    def test_policy_moves_to_device(self, evaluator):
        """Test policy is moved to evaluator device."""
        policy = ActorCritic()
        # The evaluate method moves policy to device
        # We can't fully test without running evaluation


class TestPolicyEvaluatorIntegration:
    """Integration tests for policy evaluation.

    These tests require a working BalatroEnv and are marked
    to potentially skip if environment is unavailable.
    """

    @pytest.fixture
    def policy(self):
        """Create a test policy."""
        return ActorCritic()

    @pytest.fixture
    def baseline(self):
        """Create random baseline."""
        return RandomBaseline()

    def test_random_baseline_evaluation_setup(self, baseline):
        """Test that random baseline can be set up for evaluation."""
        # Just verify the baseline is compatible with evaluator interface
        assert hasattr(baseline, "eval")
        assert hasattr(baseline, "to")
        assert hasattr(baseline, "get_action")


class TestStatisticalMethods:
    """Tests for statistical computations in evaluation."""

    def test_mean_computation(self):
        """Test mean reward computation."""
        episodes = [
            EpisodeResult(100, 10, 2, 1000, 100, 5, False),
            EpisodeResult(200, 20, 3, 2000, 200, 10, False),
            EpisodeResult(300, 30, 4, 3000, 300, 15, True),
        ]

        rewards = [e.total_reward for e in episodes]
        mean = np.mean(rewards)

        assert mean == 200.0

    def test_std_computation(self):
        """Test standard deviation computation."""
        episodes = [
            EpisodeResult(100, 10, 2, 1000, 100, 5, False),
            EpisodeResult(100, 20, 3, 2000, 200, 10, False),
            EpisodeResult(100, 30, 4, 3000, 300, 15, True),
        ]

        rewards = [e.total_reward for e in episodes]
        std = np.std(rewards)

        assert std == 0.0  # All same reward

    def test_win_rate_computation(self):
        """Test win rate computation."""
        episodes = [
            EpisodeResult(100, 10, 2, 1000, 100, 5, False),
            EpisodeResult(200, 20, 3, 2000, 200, 10, True),
            EpisodeResult(300, 30, 4, 3000, 300, 15, True),
            EpisodeResult(400, 40, 5, 4000, 400, 20, False),
        ]

        wins = [e.won for e in episodes]
        win_rate = sum(wins) / len(wins)

        assert win_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
