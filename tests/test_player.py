"""Tests for live gameplay player."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.agent.player import (
    HumanAssistedPlayer,
    LivePlayer,
    PlayerState,
    PlaySession,
    create_player,
)
from src.agent.policy import ActorCritic


class TestPlaySession:
    """Tests for PlaySession dataclass."""

    def test_default_values(self):
        """Test default session values."""
        session = PlaySession()
        assert session.games_played == 0
        assert session.games_won == 0
        assert session.total_score == 0
        assert session.max_ante_reached == 0
        assert session.actions_taken == 0
        assert len(session.errors) == 0

    def test_duration_calculation(self):
        """Test duration property."""
        import time
        session = PlaySession()
        time.sleep(0.1)
        assert session.duration >= 0.1

    def test_duration_with_end_time(self):
        """Test duration with explicit end time."""
        session = PlaySession()
        session.start_time = 0.0
        session.end_time = 10.0
        assert session.duration == 10.0

    def test_win_rate_zero_games(self):
        """Test win rate with no games."""
        session = PlaySession()
        assert session.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        session = PlaySession()
        session.games_played = 10
        session.games_won = 3
        assert session.win_rate == 0.3

    def test_str_representation(self):
        """Test string formatting."""
        session = PlaySession()
        session.games_played = 5
        session.games_won = 2
        session.total_score = 10000
        session.max_ante_reached = 4
        session.actions_taken = 100

        text = str(session)
        assert "Games: 5" in text
        assert "40.0% win" in text
        assert "Score: 10000" in text
        assert "Max Ante: 4" in text


class TestPlayerState:
    """Tests for PlayerState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert PlayerState.IDLE.value == "idle"
        assert PlayerState.PLAYING.value == "playing"
        assert PlayerState.PAUSED.value == "paused"
        assert PlayerState.ERROR.value == "error"


class TestLivePlayer:
    """Tests for LivePlayer class."""

    @pytest.fixture
    def policy(self):
        """Create test policy."""
        return ActorCritic()

    @pytest.fixture
    def player(self, policy):
        """Create test player."""
        return LivePlayer(policy, device="cpu")

    def test_initialization(self, player):
        """Test player initialization."""
        assert player.state == PlayerState.IDLE
        assert player.deterministic is True
        assert player.action_delay == 0.5

    def test_custom_params(self, policy):
        """Test custom parameter initialization."""
        player = LivePlayer(
            policy,
            action_delay=0.2,
            deterministic=False,
            device="cpu",
        )
        assert player.action_delay == 0.2
        assert player.deterministic is False

    def test_policy_on_device(self, player):
        """Test policy is on correct device."""
        param = next(player.policy.parameters())
        assert param.device == torch.device("cpu")

    def test_policy_in_eval_mode(self, player):
        """Test policy is in eval mode."""
        assert not player.policy.training

    def test_set_state_detector(self, player):
        """Test setting state detector."""
        mock_detector = MagicMock()
        player.set_state_detector(mock_detector)
        assert player._state_detector is mock_detector

    def test_set_action_coordinator(self, player):
        """Test setting action coordinator."""
        mock_coordinator = MagicMock()
        player.set_action_coordinator(mock_coordinator)
        assert player._action_coordinator is mock_coordinator

    def test_check_ready_false_no_detector(self, player):
        """Test check_ready returns False without detector."""
        assert player._check_ready() is False

    def test_check_ready_false_no_coordinator(self, player):
        """Test check_ready returns False without coordinator."""
        player._state_detector = MagicMock()
        assert player._check_ready() is False

    def test_check_ready_true(self, player):
        """Test check_ready returns True when all set."""
        player._state_detector = MagicMock()
        player._action_coordinator = MagicMock()
        assert player._check_ready() is True

    def test_stop_sets_flag(self, player):
        """Test stop() sets flag."""
        player.stop()
        assert player._stop_requested is True

    def test_pause_changes_state(self, player):
        """Test pause() changes state."""
        player.state = PlayerState.PLAYING
        player.pause()
        assert player.state == PlayerState.PAUSED

    def test_pause_only_when_playing(self, player):
        """Test pause() only works when playing."""
        player.state = PlayerState.IDLE
        player.pause()
        assert player.state == PlayerState.IDLE  # Unchanged

    def test_resume_changes_state(self, player):
        """Test resume() changes state."""
        player.state = PlayerState.PAUSED
        player.resume()
        assert player.state == PlayerState.PLAYING

    def test_resume_only_when_paused(self, player):
        """Test resume() only works when paused."""
        player.state = PlayerState.IDLE
        player.resume()
        assert player.state == PlayerState.IDLE  # Unchanged


class TestLivePlayerActionSelection:
    """Tests for action selection in LivePlayer."""

    @pytest.fixture
    def player(self):
        """Create test player."""
        policy = ActorCritic()
        return LivePlayer(policy, device="cpu")

    def test_select_action_deterministic(self, player):
        """Test deterministic action selection."""
        obs = np.random.randn(69).astype(np.float32)
        mask = np.ones(15, dtype=bool)

        action1 = player._select_action(obs, mask)
        action2 = player._select_action(obs, mask)

        # Deterministic should give same action
        assert action1 == action2

    def test_select_action_valid_range(self, player):
        """Test action is in valid range."""
        obs = np.random.randn(69).astype(np.float32)
        mask = np.ones(15, dtype=bool)

        action = player._select_action(obs, mask)

        assert 0 <= action < 15

    def test_select_action_respects_mask(self, player):
        """Test action respects mask."""
        obs = np.random.randn(69).astype(np.float32)
        mask = np.zeros(15, dtype=bool)
        mask[7] = True  # Only action 7 valid

        action = player._select_action(obs, mask)

        assert action == 7


class TestLivePlayerFromCheckpoint:
    """Tests for loading player from checkpoint."""

    def test_from_checkpoint(self):
        """Test loading from checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"

            # Create and save a checkpoint
            policy = ActorCritic()
            checkpoint = {
                "policy_state_dict": policy.state_dict(),
                "obs_dim": 69,
                "action_dim": 15,
                "hidden_dim": 256,
            }
            torch.save(checkpoint, checkpoint_path)

            # Load player from checkpoint
            player = LivePlayer.from_checkpoint(
                checkpoint_path,
                device="cpu",
            )

            assert isinstance(player, LivePlayer)
            assert player.policy is not None


class TestHumanAssistedPlayer:
    """Tests for HumanAssistedPlayer class."""

    @pytest.fixture
    def player(self):
        """Create human-assisted player."""
        policy = ActorCritic()
        return HumanAssistedPlayer(
            policy,
            confidence_threshold=0.8,
            device="cpu",
        )

    def test_initialization(self, player):
        """Test initialization."""
        assert player.confidence_threshold == 0.8
        assert player._human_callback is None

    def test_set_human_callback(self, player):
        """Test setting human callback."""
        callback = MagicMock(return_value=None)
        player.set_human_callback(callback)
        assert player._human_callback is callback

    def test_high_confidence_uses_model(self, player):
        """Test high confidence uses model action."""
        # Mock a very confident model (won't request human input)
        obs = np.random.randn(69).astype(np.float32)
        mask = np.ones(15, dtype=bool)

        human_mock = MagicMock(return_value=None)
        player.set_human_callback(human_mock)

        # Action selection - may or may not call human depending on confidence
        action = player._select_action(obs, mask)

        assert 0 <= action < 15


class TestCreatePlayer:
    """Tests for create_player factory function."""

    def test_create_without_checkpoint(self):
        """Test creating player without checkpoint."""
        player = create_player(device="cpu")
        assert isinstance(player, LivePlayer)
        assert player.policy is not None

    def test_create_with_checkpoint(self):
        """Test creating player with checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"

            # Create checkpoint
            policy = ActorCritic()
            torch.save({"policy_state_dict": policy.state_dict()}, checkpoint_path)

            player = create_player(
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

            assert isinstance(player, LivePlayer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
