"""Live gameplay player for Balatro RL agent.

Connects a trained policy to the actual game through
the vision and executor modules for autonomous play.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from src.agent.policy import ActorCritic

if TYPE_CHECKING:
    from src.executor.coordinator import ActionCoordinator
    from src.vision.state_detector import StateDetector

logger = logging.getLogger(__name__)


class PlayerState(Enum):
    """Player operational states."""

    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class PlaySession:
    """Statistics from a play session.

    Attributes:
        start_time: Session start timestamp.
        end_time: Session end timestamp.
        games_played: Number of games completed.
        games_won: Number of games won.
        total_score: Cumulative score across games.
        max_ante_reached: Highest ante achieved.
        actions_taken: Total actions executed.
        errors: List of errors encountered.
    """

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    games_played: int = 0
    games_won: int = 0
    total_score: int = 0
    max_ante_reached: int = 0
    actions_taken: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def win_rate(self) -> float:
        """Win rate as fraction."""
        if self.games_played == 0:
            return 0.0
        return self.games_won / self.games_played

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"PlaySession ({self.duration:.0f}s):\n"
            f"  Games: {self.games_played} ({self.win_rate*100:.1f}% win)\n"
            f"  Score: {self.total_score}\n"
            f"  Max Ante: {self.max_ante_reached}\n"
            f"  Actions: {self.actions_taken}"
        )


class LivePlayer:
    """Plays Balatro using a trained policy.

    Connects the RL policy to vision (for state observation)
    and executor (for action execution) to play autonomously.

    Example:
        player = LivePlayer.from_checkpoint("models/best.pt")
        player.play(num_games=10)
    """

    def __init__(
        self,
        policy: ActorCritic,
        state_detector: StateDetector | None = None,
        action_coordinator: ActionCoordinator | None = None,
        device: str = "auto",
        action_delay: float = 0.5,
        deterministic: bool = True,
    ) -> None:
        """Initialize live player.

        Args:
            policy: Trained policy network.
            state_detector: Vision system for state observation.
            action_coordinator: Executor for action execution.
            device: Device for policy inference.
            action_delay: Delay between actions (seconds).
            deterministic: Use deterministic actions (argmax).
        """
        self.policy = policy
        self._state_detector = state_detector
        self._action_coordinator = action_coordinator
        self.action_delay = action_delay
        self.deterministic = deterministic

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy.to(self.device)
        self.policy.eval()

        self.state = PlayerState.IDLE
        self._session: PlaySession | None = None
        self._stop_requested = False

        logger.info(f"LivePlayer initialized on {self.device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        state_detector: StateDetector | None = None,
        action_coordinator: ActionCoordinator | None = None,
        device: str = "auto",
        **kwargs,
    ) -> "LivePlayer":
        """Create player from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            state_detector: Vision system.
            action_coordinator: Executor system.
            device: Device for inference.
            **kwargs: Additional arguments for LivePlayer.

        Returns:
            Initialized LivePlayer.
        """
        checkpoint_path = Path(checkpoint_path)

        if device == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            map_location = device

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Extract model config if available
        obs_dim = checkpoint.get("obs_dim", 69)
        action_dim = checkpoint.get("action_dim", 15)
        hidden_dim = checkpoint.get("hidden_dim", 256)

        policy = ActorCritic(obs_dim, action_dim, hidden_dim)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return cls(
            policy=policy,
            state_detector=state_detector,
            action_coordinator=action_coordinator,
            device=device,
            **kwargs,
        )

    def set_state_detector(self, detector: StateDetector) -> None:
        """Set the state detector (vision system).

        Args:
            detector: StateDetector instance.
        """
        self._state_detector = detector

    def set_action_coordinator(self, coordinator: ActionCoordinator) -> None:
        """Set the action coordinator (executor system).

        Args:
            coordinator: ActionCoordinator instance.
        """
        self._action_coordinator = coordinator

    def _check_ready(self) -> bool:
        """Check if player is ready to play.

        Returns:
            True if all components are set.
        """
        if self._state_detector is None:
            logger.error("StateDetector not set")
            return False
        if self._action_coordinator is None:
            logger.error("ActionCoordinator not set")
            return False
        return True

    def _observe_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Observe current game state.

        Returns:
            Tuple of (observation, action_mask).
        """
        # Get screen capture and detect state
        state = self._state_detector.detect()

        # Convert to observation vector (this would use the actual
        # state encoding from BalatroEnv - simplified here)
        obs = self._encode_observation(state)

        # Get valid actions from state
        action_mask = self._get_action_mask(state)

        return obs, action_mask

    def _encode_observation(self, state) -> np.ndarray:
        """Encode game state to observation vector.

        This should match the encoding used during training
        in BalatroEnv._get_observation().

        Args:
            state: Detected game state.

        Returns:
            Observation vector.
        """
        # Placeholder - actual implementation would convert
        # the detected state to the same format used in training
        obs = np.zeros(69, dtype=np.float32)

        # Example encoding (would be more detailed in practice):
        # - Cards in hand
        # - Current score
        # - Money
        # - Ante/blind info
        # - Joker info

        return obs

    def _get_action_mask(self, state) -> np.ndarray:
        """Get mask of valid actions for current state.

        Args:
            state: Detected game state.

        Returns:
            Boolean mask where True = valid action.
        """
        # Default: all actions valid (would be refined based on state)
        mask = np.ones(15, dtype=bool)
        return mask

    def _select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Select action using policy.

        Args:
            obs: Observation vector.
            action_mask: Valid action mask.

        Returns:
            Selected action index.
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

            action, _, _ = self.policy.get_action(
                obs_tensor, mask_tensor, deterministic=self.deterministic
            )

            return action.item()

    def _execute_action(self, action: int) -> bool:
        """Execute action in game.

        Args:
            action: Action index to execute.

        Returns:
            True if execution succeeded.
        """
        try:
            self._action_coordinator.execute(action)
            return True
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    def play_step(self) -> bool:
        """Execute a single play step.

        Returns:
            True if step succeeded.
        """
        if not self._check_ready():
            return False

        try:
            # Observe
            obs, action_mask = self._observe_state()

            # Decide
            action = self._select_action(obs, action_mask)

            # Act
            success = self._execute_action(action)

            if success and self._session:
                self._session.actions_taken += 1

            # Delay for game to update
            time.sleep(self.action_delay)

            return success

        except Exception as e:
            logger.error(f"Play step error: {e}")
            if self._session:
                self._session.errors.append(str(e))
            return False

    def play_game(
        self,
        max_steps: int = 1000,
        callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Play a single game.

        Args:
            max_steps: Maximum steps before stopping.
            callback: Optional callback(step, action) per step.

        Returns:
            Game statistics dict.
        """
        if not self._check_ready():
            return {"error": "Player not ready"}

        self.state = PlayerState.PLAYING
        steps = 0
        errors = 0

        logger.info("Starting game")

        while steps < max_steps and not self._stop_requested:
            if self.state == PlayerState.PAUSED:
                time.sleep(0.1)
                continue

            success = self.play_step()

            if success:
                steps += 1
                if callback:
                    callback(steps, 0)  # Would pass actual action
            else:
                errors += 1
                if errors > 10:
                    logger.error("Too many errors, stopping game")
                    break

            # Check if game ended (would check via state detector)
            # Placeholder: actual implementation would detect game over

        self.state = PlayerState.IDLE

        return {
            "steps": steps,
            "errors": errors,
            "completed": not self._stop_requested,
        }

    def play(
        self,
        num_games: int = 1,
        max_steps_per_game: int = 1000,
        callback: Callable[[int, dict], None] | None = None,
    ) -> PlaySession:
        """Play multiple games.

        Args:
            num_games: Number of games to play.
            max_steps_per_game: Maximum steps per game.
            callback: Optional callback(game_num, stats) per game.

        Returns:
            PlaySession with aggregated statistics.
        """
        if not self._check_ready():
            session = PlaySession()
            session.errors.append("Player not ready")
            return session

        self._session = PlaySession()
        self._stop_requested = False

        logger.info(f"Starting play session: {num_games} games")

        for game_num in range(num_games):
            if self._stop_requested:
                break

            logger.info(f"Game {game_num + 1}/{num_games}")

            game_result = self.play_game(max_steps=max_steps_per_game)

            self._session.games_played += 1

            if callback:
                callback(game_num + 1, game_result)

            # Brief pause between games
            time.sleep(1.0)

        self._session.end_time = time.time()
        logger.info(f"Session complete: {self._session}")

        return self._session

    def stop(self) -> None:
        """Request player to stop."""
        logger.info("Stop requested")
        self._stop_requested = True

    def pause(self) -> None:
        """Pause the player."""
        if self.state == PlayerState.PLAYING:
            self.state = PlayerState.PAUSED
            logger.info("Player paused")

    def resume(self) -> None:
        """Resume the player."""
        if self.state == PlayerState.PAUSED:
            self.state = PlayerState.PLAYING
            logger.info("Player resumed")


class HumanAssistedPlayer(LivePlayer):
    """Player that can request human input for uncertain decisions.

    Useful for debugging or semi-automated play where the
    model may need guidance on edge cases.
    """

    def __init__(
        self,
        policy: ActorCritic,
        confidence_threshold: float = 0.8,
        **kwargs,
    ) -> None:
        """Initialize human-assisted player.

        Args:
            policy: Trained policy network.
            confidence_threshold: Request help below this confidence.
            **kwargs: Additional LivePlayer arguments.
        """
        super().__init__(policy, **kwargs)
        self.confidence_threshold = confidence_threshold
        self._human_callback: Callable[[np.ndarray, np.ndarray], int | None] | None = None

    def set_human_callback(
        self,
        callback: Callable[[np.ndarray, np.ndarray], int | None],
    ) -> None:
        """Set callback for human input.

        Args:
            callback: Function(obs, action_mask) -> action or None.
        """
        self._human_callback = callback

    def _select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Select action, possibly with human assistance.

        Args:
            obs: Observation vector.
            action_mask: Valid action mask.

        Returns:
            Selected action index.
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

            logits, _ = self.policy(obs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)
            max_prob = probs.max().item()

            if max_prob < self.confidence_threshold and self._human_callback:
                # Request human input
                human_action = self._human_callback(obs, action_mask)
                if human_action is not None:
                    logger.info(f"Using human action: {human_action}")
                    return human_action

            # Use model action
            action, _, _ = self.policy.get_action(
                obs_tensor, mask_tensor, deterministic=self.deterministic
            )

            return action.item()


def create_player(
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    **kwargs,
) -> LivePlayer:
    """Create a LivePlayer, optionally loading a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint (or None for fresh policy).
        device: Device for inference.
        **kwargs: Additional arguments.

    Returns:
        Initialized LivePlayer.
    """
    if checkpoint_path:
        return LivePlayer.from_checkpoint(checkpoint_path, device=device, **kwargs)
    else:
        policy = ActorCritic()
        return LivePlayer(policy, device=device, **kwargs)
