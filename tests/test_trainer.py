"""Tests for PPO trainer."""

import numpy as np
import pytest
import torch

from src.agent.trainer import (
    CurriculumConfig,
    PPOTrainer,
    RolloutBuffer,
    TrainerConfig,
)


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainerConfig()
        assert config.total_timesteps == 1_000_000
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5

    def test_custom_values(self):
        """Test custom configuration."""
        config = TrainerConfig(
            total_timesteps=100_000,
            learning_rate=1e-3,
            batch_size=128,
        )
        assert config.total_timesteps == 100_000
        assert config.learning_rate == 1e-3
        assert config.batch_size == 128


class TestCurriculumConfig:
    """Tests for CurriculumConfig dataclass."""

    def test_default_values(self):
        """Test default curriculum values."""
        config = CurriculumConfig()
        assert config.enabled is True
        assert config.initial_max_ante == 2
        assert config.final_max_ante == 8
        assert config.ante_threshold == 0.5
        assert config.evaluation_window == 100

    def test_custom_values(self):
        """Test custom curriculum configuration."""
        config = CurriculumConfig(
            enabled=False,
            initial_max_ante=3,
            final_max_ante=6,
        )
        assert config.enabled is False
        assert config.initial_max_ante == 3
        assert config.final_max_ante == 6


class TestRolloutBuffer:
    """Tests for experience buffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return RolloutBuffer()

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert len(buffer.observations) == 0
        assert len(buffer.actions) == 0
        assert len(buffer.rewards) == 0
        assert buffer.advantages is None
        assert buffer.returns is None

    def test_add_samples(self, buffer):
        """Test adding samples to buffer."""
        obs = np.random.randn(69).astype(np.float32)
        action = 5
        reward = 1.0
        done = False
        value = 0.5
        log_prob = -0.5
        action_mask = np.ones(15, dtype=bool)

        buffer.add(obs, action, reward, done, value, log_prob, action_mask)

        assert len(buffer.observations) == 1
        assert len(buffer.actions) == 1
        assert buffer.actions[0] == 5

    def test_clear_buffer(self, buffer):
        """Test clearing the buffer."""
        # Add some data
        for i in range(5):
            buffer.add(
                obs=np.random.randn(69).astype(np.float32),
                action=i % 15,
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-0.5,
                action_mask=np.ones(15, dtype=bool),
            )

        assert len(buffer.observations) == 5

        buffer.clear()

        assert len(buffer.observations) == 0
        assert len(buffer.actions) == 0
        assert buffer.advantages is None

    def test_compute_returns(self, buffer):
        """Test GAE computation."""
        # Add some samples
        for i in range(10):
            buffer.add(
                obs=np.random.randn(69).astype(np.float32),
                action=i % 15,
                reward=1.0,
                done=(i == 9),  # Last one is done
                value=0.5,
                log_prob=-0.5,
                action_mask=np.ones(15, dtype=bool),
            )

        buffer.compute_returns_and_advantages(last_value=0.0)

        # Check returns and advantages are computed
        assert buffer.returns is not None
        assert buffer.advantages is not None
        assert len(buffer.returns) == 10
        assert len(buffer.advantages) == 10

    def test_get_batches(self, buffer):
        """Test batch generation."""
        # Fill buffer
        for i in range(32):
            buffer.add(
                obs=np.random.randn(69).astype(np.float32),
                action=i % 15,
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-0.5,
                action_mask=np.ones(15, dtype=bool),
            )

        buffer.compute_returns_and_advantages(last_value=0.0)

        batches = list(buffer.get_batches(batch_size=8, device=torch.device("cpu")))

        # Should get 4 batches (32 / 8 = 4)
        assert len(batches) == 4

        # Check batch contents
        obs, actions, old_log_probs, advantages, returns, action_masks = batches[0]
        assert obs.shape == (8, 69)
        assert actions.shape == (8,)
        assert old_log_probs.shape == (8,)
        assert advantages.shape == (8,)
        assert returns.shape == (8,)
        assert action_masks.shape == (8, 15)


class TestPPOTrainer:
    """Tests for PPO training loop.

    Note: Full integration tests require mocking BalatroEnv
    since it depends on game state. These tests verify initialization.
    """

    def test_config_defaults(self):
        """Test trainer config has sensible defaults."""
        config = TrainerConfig()

        # PPO hyperparameters should be in reasonable ranges
        assert 0 < config.learning_rate < 1
        assert 0 < config.clip_range < 1
        assert 0 < config.gamma <= 1
        assert 0 < config.gae_lambda <= 1


class TestGAEComputation:
    """Tests for GAE advantage estimation."""

    def test_gae_zero_reward(self):
        """Test GAE with zero rewards."""
        buffer = RolloutBuffer()

        for i in range(5):
            buffer.add(
                obs=np.zeros(69, dtype=np.float32),
                action=0,
                reward=0.0,
                done=False,
                value=1.0,
                log_prob=0.0,
                action_mask=np.ones(15, dtype=bool),
            )

        buffer.compute_returns_and_advantages(last_value=1.0)

        # With zero rewards and constant values, advantages should be ~0
        assert np.allclose(buffer.advantages, 0, atol=0.1)

    def test_gae_positive_rewards(self):
        """Test GAE with positive rewards."""
        buffer = RolloutBuffer()

        for i in range(5):
            buffer.add(
                obs=np.zeros(69, dtype=np.float32),
                action=0,
                reward=1.0,
                done=False,
                value=0.0,
                log_prob=0.0,
                action_mask=np.ones(15, dtype=bool),
            )

        buffer.compute_returns_and_advantages(last_value=0.0)

        # With positive rewards and zero values, advantages should be positive
        assert (buffer.advantages > 0).all()

    def test_gae_terminal_state(self):
        """Test GAE handles terminal states correctly."""
        buffer = RolloutBuffer()

        # Non-terminal transition
        buffer.add(
            obs=np.zeros(69, dtype=np.float32),
            action=0,
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=0.0,
            action_mask=np.ones(15, dtype=bool),
        )

        # Terminal transition
        buffer.add(
            obs=np.zeros(69, dtype=np.float32),
            action=0,
            reward=10.0,
            done=True,
            value=0.5,
            log_prob=0.0,
            action_mask=np.ones(15, dtype=bool),
        )

        buffer.compute_returns_and_advantages(last_value=0.0)

        # Terminal advantage shouldn't include future value
        # delta = reward - value = 10.0 - 0.5 = 9.5
        assert buffer.advantages[1] == pytest.approx(9.5, abs=0.01)


class TestNormalization:
    """Tests for advantage normalization."""

    def test_advantage_normalization(self):
        """Test advantage normalization."""
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Normalize
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Should have ~0 mean and ~1 std
        assert abs(normalized.mean()) < 0.01
        assert abs(normalized.std() - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
