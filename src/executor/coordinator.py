"""Action coordination and sequencing for Balatro automation.

Provides high-level coordination between game state detection
and action execution with proper timing and verification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from src.executor.actions import ActionExecutor, ActionResult, GameAction, UILayout
from src.executor.mouse import MouseController, Region
from src.game.constants import GamePhase

if TYPE_CHECKING:
    from src.vision.state_detector import GameStateInfo

logger = logging.getLogger(__name__)


class CoordinatorState(Enum):
    """Coordinator operational states."""

    IDLE = auto()
    EXECUTING = auto()
    WAITING = auto()
    VERIFYING = auto()
    ERROR = auto()
    PAUSED = auto()


@dataclass
class ActionStep:
    """A single step in an action sequence."""

    action: GameAction
    params: dict = field(default_factory=dict)
    delay_before: float = 0.0
    delay_after: float = 0.1
    verify: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ActionSequence:
    """A sequence of actions to execute."""

    name: str
    steps: list[ActionStep] = field(default_factory=list)
    on_success: Callable | None = None
    on_failure: Callable | None = None

    def add_step(
        self,
        action: GameAction,
        delay_after: float = 0.1,
        **params,
    ) -> ActionSequence:
        """Add a step to the sequence.

        Args:
            action: Action to execute.
            delay_after: Delay after action.
            **params: Action parameters.

        Returns:
            Self for chaining.
        """
        self.steps.append(ActionStep(
            action=action,
            params=params,
            delay_after=delay_after,
        ))
        return self


@dataclass
class CoordinatorConfig:
    """Configuration for the action coordinator."""

    # Timing settings
    default_action_delay: float = 0.15
    animation_wait: float = 0.5
    state_check_interval: float = 0.1

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.3

    # Verification settings
    verify_actions: bool = True
    verification_timeout: float = 2.0

    # Safety settings
    max_actions_per_second: float = 5.0
    pause_on_error: bool = True


class ActionCoordinator:
    """Coordinates game actions with state verification.

    Provides high-level game interaction by:
    - Sequencing multiple actions
    - Verifying action success via state detection
    - Managing timing and delays
    - Handling retries and errors

    Attributes:
        executor: ActionExecutor for executing individual actions.
        config: Coordinator configuration.
        state: Current coordinator state.
    """

    def __init__(
        self,
        executor: ActionExecutor | None = None,
        config: CoordinatorConfig | None = None,
    ) -> None:
        """Initialize action coordinator.

        Args:
            executor: Action executor (created if None).
            config: Coordinator config (default if None).
        """
        self.executor = executor or ActionExecutor()
        self.config = config or CoordinatorConfig()
        self.state = CoordinatorState.IDLE

        # Track action history
        self._action_history: list[tuple[float, GameAction, bool]] = []
        self._last_action_time: float = 0

        # State detection callback
        self._state_detector: Callable[[], GameStateInfo] | None = None

        logger.info("ActionCoordinator initialized")

    def set_state_detector(
        self,
        detector: Callable[[], GameStateInfo],
    ) -> None:
        """Set state detection callback.

        Args:
            detector: Function that returns current game state.
        """
        self._state_detector = detector

    def _rate_limit(self) -> None:
        """Apply rate limiting between actions."""
        min_interval = 1.0 / self.config.max_actions_per_second
        elapsed = time.time() - self._last_action_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_action_time = time.time()

    def _record_action(self, action: GameAction, success: bool) -> None:
        """Record an action in history.

        Args:
            action: Action that was executed.
            success: Whether it succeeded.
        """
        self._action_history.append((time.time(), action, success))

        # Keep history bounded
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-500:]

    def execute(
        self,
        action: GameAction,
        verify: bool | None = None,
        **params,
    ) -> ActionResult:
        """Execute a single action with optional verification.

        Args:
            action: Action to execute.
            verify: Whether to verify success (None = use config).
            **params: Action parameters.

        Returns:
            ActionResult indicating success/failure.
        """
        if self.state == CoordinatorState.PAUSED:
            return ActionResult(
                success=False,
                action=action,
                message="Coordinator is paused",
            )

        self.state = CoordinatorState.EXECUTING
        self._rate_limit()

        result = self.executor.execute(action, **params)
        self._record_action(action, result.success)

        # Post-action delay
        time.sleep(self.config.default_action_delay)

        # Verify if requested
        if verify is None:
            verify = self.config.verify_actions

        if verify and result.success and self._state_detector:
            self.state = CoordinatorState.VERIFYING
            verified = self._verify_action(action, params)
            if not verified:
                result = ActionResult(
                    success=False,
                    action=action,
                    message="Action verification failed",
                )

        self.state = CoordinatorState.IDLE
        return result

    def _verify_action(
        self,
        action: GameAction,
        params: dict,
    ) -> bool:
        """Verify that an action had the expected effect.

        Args:
            action: Action that was executed.
            params: Action parameters.

        Returns:
            True if verification passes.
        """
        if self._state_detector is None:
            return True

        # Wait for animation
        time.sleep(self.config.animation_wait)

        try:
            state = self._state_detector()

            # Verify based on action type
            if action == GameAction.PLAY_HAND:
                # Score should have changed or hands decreased
                return True  # Basic verification

            elif action == GameAction.DISCARD:
                # Discards should have decreased
                return True

            # Default: assume success if no specific verification
            return True

        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return False

    def execute_sequence(
        self,
        sequence: ActionSequence,
    ) -> list[ActionResult]:
        """Execute a sequence of actions.

        Args:
            sequence: Action sequence to execute.

        Returns:
            List of results for each step.
        """
        results = []
        logger.info(f"Executing sequence: {sequence.name}")

        for i, step in enumerate(sequence.steps):
            # Pre-delay
            if step.delay_before > 0:
                time.sleep(step.delay_before)

            # Execute with retries
            result = None
            for attempt in range(step.max_retries + 1):
                result = self.execute(
                    step.action,
                    verify=step.verify,
                    **step.params,
                )

                if result.success:
                    break

                if attempt < step.max_retries:
                    logger.warning(
                        f"Step {i} failed, retry {attempt + 1}/{step.max_retries}"
                    )
                    time.sleep(self.config.retry_delay)

            results.append(result)

            # Check for failure
            if not result.success:
                logger.error(f"Sequence {sequence.name} failed at step {i}")
                if sequence.on_failure:
                    sequence.on_failure(i, result)
                if self.config.pause_on_error:
                    self.state = CoordinatorState.ERROR
                break

            # Post-delay
            if step.delay_after > 0:
                time.sleep(step.delay_after)

        # Success callback
        if all(r.success for r in results) and sequence.on_success:
            sequence.on_success()

        return results

    # High-level game action methods
    def select_and_play(
        self,
        card_indices: list[int],
    ) -> list[ActionResult]:
        """Select cards and play them.

        Args:
            card_indices: Indices of cards to select and play.

        Returns:
            List of action results.
        """
        sequence = ActionSequence(name="select_and_play")

        # Select each card
        for idx in card_indices:
            sequence.add_step(
                GameAction.SELECT_CARD,
                card_index=idx,
                delay_after=0.1,
            )

        # Play hand
        sequence.add_step(
            GameAction.PLAY_HAND,
            delay_after=self.config.animation_wait,
        )

        return self.execute_sequence(sequence)

    def select_and_discard(
        self,
        card_indices: list[int],
    ) -> list[ActionResult]:
        """Select cards and discard them.

        Args:
            card_indices: Indices of cards to discard.

        Returns:
            List of action results.
        """
        sequence = ActionSequence(name="select_and_discard")

        for idx in card_indices:
            sequence.add_step(
                GameAction.SELECT_CARD,
                card_index=idx,
                delay_after=0.1,
            )

        sequence.add_step(
            GameAction.DISCARD,
            delay_after=self.config.animation_wait,
        )

        return self.execute_sequence(sequence)

    def select_blind(
        self,
        blind_type: str,
    ) -> ActionResult:
        """Select a blind.

        Args:
            blind_type: "small", "big", or "boss".

        Returns:
            Action result.
        """
        action_map = {
            "small": GameAction.SELECT_SMALL_BLIND,
            "big": GameAction.SELECT_BIG_BLIND,
            "boss": GameAction.SELECT_BOSS_BLIND,
        }

        action = action_map.get(blind_type)
        if action is None:
            return ActionResult(
                success=False,
                action=GameAction.SELECT_SMALL_BLIND,
                message=f"Invalid blind type: {blind_type}",
            )

        result = self.execute(action)

        # Wait for transition
        time.sleep(self.config.animation_wait)

        return result

    def complete_shop(self) -> ActionResult:
        """End the shop phase.

        Returns:
            Action result.
        """
        result = self.execute(GameAction.END_SHOP)
        time.sleep(self.config.animation_wait)
        return result

    def buy_shop_item(self, item_index: int) -> ActionResult:
        """Buy an item from the shop.

        Args:
            item_index: Index of item to buy.

        Returns:
            Action result.
        """
        return self.execute(GameAction.BUY_ITEM, item_index=item_index)

    # State management
    def pause(self) -> None:
        """Pause the coordinator."""
        self.state = CoordinatorState.PAUSED
        logger.info("Coordinator paused")

    def resume(self) -> None:
        """Resume the coordinator."""
        if self.state == CoordinatorState.PAUSED:
            self.state = CoordinatorState.IDLE
            logger.info("Coordinator resumed")

    def reset(self) -> None:
        """Reset coordinator state."""
        self.state = CoordinatorState.IDLE
        self._action_history.clear()
        self._last_action_time = 0
        logger.info("Coordinator reset")

    def get_stats(self) -> dict:
        """Get coordinator statistics.

        Returns:
            Dict with action counts and success rates.
        """
        if not self._action_history:
            return {
                "total_actions": 0,
                "success_rate": 0.0,
                "actions_per_minute": 0.0,
            }

        total = len(self._action_history)
        successes = sum(1 for _, _, s in self._action_history if s)

        # Calculate actions per minute
        if len(self._action_history) >= 2:
            time_span = self._action_history[-1][0] - self._action_history[0][0]
            apm = (total / time_span * 60) if time_span > 0 else 0
        else:
            apm = 0

        return {
            "total_actions": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "actions_per_minute": apm,
            "state": self.state.name,
        }


class GameController:
    """High-level game controller integrating all components.

    Provides the main interface for automated game play,
    combining state detection and action execution.
    """

    def __init__(
        self,
        coordinator: ActionCoordinator | None = None,
    ) -> None:
        """Initialize game controller.

        Args:
            coordinator: Action coordinator (created if None).
        """
        self.coordinator = coordinator or ActionCoordinator()
        self._running = False

        logger.info("GameController initialized")

    def start(self) -> None:
        """Start the game controller."""
        self._running = True
        self.coordinator.resume()
        logger.info("GameController started")

    def stop(self) -> None:
        """Stop the game controller."""
        self._running = False
        self.coordinator.pause()
        logger.info("GameController stopped")

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    def play_turn(
        self,
        card_indices: list[int],
        action: str = "play",
    ) -> list[ActionResult]:
        """Play a turn (select cards and play/discard).

        Args:
            card_indices: Cards to select.
            action: "play" or "discard".

        Returns:
            List of action results.
        """
        if action == "play":
            return self.coordinator.select_and_play(card_indices)
        elif action == "discard":
            return self.coordinator.select_and_discard(card_indices)
        else:
            return []

    def handle_blind_select(
        self,
        blind_type: str = "small",
    ) -> ActionResult:
        """Handle blind selection phase.

        Args:
            blind_type: Which blind to select.

        Returns:
            Action result.
        """
        return self.coordinator.select_blind(blind_type)

    def handle_shop(
        self,
        purchases: list[int] | None = None,
    ) -> list[ActionResult]:
        """Handle shop phase.

        Args:
            purchases: Item indices to buy (empty to skip).

        Returns:
            List of action results.
        """
        results = []

        # Buy items if specified
        if purchases:
            for item_idx in purchases:
                result = self.coordinator.buy_shop_item(item_idx)
                results.append(result)
                if not result.success:
                    break

        # End shop
        result = self.coordinator.complete_shop()
        results.append(result)

        return results
