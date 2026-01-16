"""Gymnasium environment for Balatro RL training.

Provides a standard Gymnasium interface for training RL agents on Balatro.
Supports both discrete action spaces and action masking for invalid actions.
"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.constants import (
    ActionType,
    Enhancement,
    GamePhase,
    Rank,
    Suit,
    MAX_HAND_SIZE,
    STARTING_HANDS,
    STARTING_DISCARDS,
)
from src.game.state_machine import GameStateMachine, GameState

logger = logging.getLogger(__name__)


class BalatroEnv(gym.Env):
    """Gymnasium environment for Balatro.

    Observation space includes:
    - Game phase (one-hot)
    - Ante, score, money, hands, discards (normalized)
    - Hand cards (encoded as rank + suit)
    - Selected cards mask

    Action space:
    - 0: Play selected hand
    - 1: Discard selected
    - 2-9: Toggle card selection (cards 0-7)
    - 10: Select blind (small)
    - 11: Select blind (big)
    - 12: Select blind (boss)
    - 13: Skip blind
    - 14: End shop

    Attributes:
        game: GameStateMachine instance.
        max_ante: Maximum ante for curriculum learning.
        render_mode: Rendering mode (None or "human").
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        max_ante: int = 8,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            max_ante: Maximum ante to play to (for curriculum).
            render_mode: Rendering mode.
            seed: Random seed.
        """
        super().__init__()

        self.max_ante = max_ante
        self.render_mode = render_mode
        self._seed = seed

        self.game = GameStateMachine(seed=seed)

        # Action space: 15 discrete actions
        # 0: play, 1: discard, 2-9: toggle cards, 10-12: select blinds, 13: skip, 14: end shop
        self.action_space = spaces.Discrete(15)

        # Observation space
        # Phase (7), ante (1), score (1), chips_needed (1), money (1),
        # hands (1), discards (1), hand_cards (8 * 6), selected (8)
        obs_size = 7 + 6 + 8 * 6 + 8  # = 69
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Track episode stats
        self._episode_score = 0
        self._episode_ante = 1

        logger.info(f"BalatroEnv initialized, max_ante={max_ante}")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self.game = GameStateMachine(seed=seed)

        self.game.new_game()
        self._episode_score = 0
        self._episode_ante = 1

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take (0-14).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        state = self.game.state
        reward = 0.0
        terminated = False
        truncated = False

        # Execute action based on current phase
        if action == 0:  # Play hand
            if state.phase == GamePhase.PLAYING and state.hands_remaining > 0:
                old_score = state.score
                self.game.play_hand()
                reward = self._calculate_play_reward(old_score)

        elif action == 1:  # Discard
            if state.phase == GamePhase.PLAYING and state.discards_remaining > 0:
                self.game.discard()
                reward = -0.1  # Small penalty for discarding

        elif 2 <= action <= 9:  # Toggle card selection
            card_idx = action - 2
            if state.phase == GamePhase.PLAYING and card_idx < len(state.hand):
                card = state.hand[card_idx]
                if card in state.played_cards:
                    state.played_cards.remove(card)
                else:
                    if len(state.played_cards) < 5:
                        state.played_cards.append(card)

        elif action == 10:  # Select small blind
            if state.phase == GamePhase.BLIND_SELECT:
                self.game.select_blind("small")

        elif action == 11:  # Select big blind
            if state.phase == GamePhase.BLIND_SELECT:
                self.game.select_blind("big")

        elif action == 12:  # Select boss blind
            if state.phase == GamePhase.BLIND_SELECT:
                self.game.select_blind("boss")

        elif action == 13:  # Skip blind
            if state.phase == GamePhase.BLIND_SELECT:
                self.game.skip_blind()

        elif action == 14:  # End shop
            if state.phase == GamePhase.SHOP:
                self.game.end_shop()

        # Check termination
        if state.phase == GamePhase.GAME_OVER:
            terminated = True
            if state.ante > self._episode_ante:
                # Bonus for reaching new ante
                reward += (state.ante - self._episode_ante) * 10
            self._episode_ante = state.ante

        # Truncate if exceeded max ante (curriculum)
        if state.ante > self.max_ante:
            truncated = True
            reward += 50  # Bonus for completing curriculum

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _calculate_play_reward(self, old_score: int) -> float:
        """Calculate reward for playing a hand.

        Args:
            old_score: Score before playing.

        Returns:
            Reward value.
        """
        state = self.game.state
        score_gained = state.score - old_score

        if state.blind:
            chips_needed = state.blind.chips_required
            progress = score_gained / max(chips_needed, 1)

            # Reward for progress toward blind
            reward = progress * 10

            # Bonus for beating blind
            if state.score >= chips_needed:
                reward += 20

                # Extra bonus for efficiency (fewer hands used)
                hands_used = STARTING_HANDS - state.hands_remaining
                efficiency_bonus = max(0, (STARTING_HANDS - hands_used)) * 2
                reward += efficiency_bonus

            return reward

        return score_gained / 1000  # Fallback

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation as numpy array.
        """
        state = self.game.state
        obs = []

        # Phase one-hot (7 values)
        phase_onehot = [0.0] * 7
        phase_idx = min(state.phase.value, 6)
        phase_onehot[phase_idx] = 1.0
        obs.extend(phase_onehot)

        # Normalized values
        obs.append(state.ante / 8.0)
        obs.append(min(state.score / 100000, 1.0))
        chips_needed = state.blind.chips_required if state.blind else 1
        obs.append(min(chips_needed / 100000, 1.0))
        obs.append(min(state.money / 100, 1.0))
        obs.append(state.hands_remaining / STARTING_HANDS)
        obs.append(state.discards_remaining / STARTING_DISCARDS)

        # Suit index mapping
        suit_indices = {Suit.HEARTS: 0, Suit.DIAMONDS: 1, Suit.CLUBS: 2, Suit.SPADES: 3}

        # Hand cards (8 cards * 6 features each)
        for i in range(MAX_HAND_SIZE):
            if i < len(state.hand):
                card = state.hand[i]
                # Rank normalized (2-14 -> 0-1)
                obs.append((card.rank.value - 2) / 12)
                # Suit one-hot (4 values)
                suit_onehot = [0.0] * 4
                suit_onehot[suit_indices[card.suit]] = 1.0
                obs.extend(suit_onehot)
                # Enhancement indicator
                obs.append(1.0 if card.enhancement != Enhancement.NONE else 0.0)
            else:
                # Empty card slot
                obs.extend([0.0] * 6)

        # Selected cards mask (8 values)
        for i in range(MAX_HAND_SIZE):
            if i < len(state.hand) and state.hand[i] in state.played_cards:
                obs.append(1.0)
            else:
                obs.append(0.0)

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        """Get info dictionary.

        Returns:
            Info dict with game state details.
        """
        state = self.game.state
        return {
            "ante": state.ante,
            "phase": state.phase.name,
            "score": state.score,
            "money": state.money,
            "hands_remaining": state.hands_remaining,
            "discards_remaining": state.discards_remaining,
            "valid_actions": self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean array where True = action is valid.
        """
        state = self.game.state
        mask = np.zeros(15, dtype=bool)

        if state.phase == GamePhase.PLAYING:
            # Can play if cards selected and hands remaining
            if state.played_cards and state.hands_remaining > 0:
                mask[0] = True
            # Can discard if cards selected and discards remaining
            if state.played_cards and state.discards_remaining > 0:
                mask[1] = True
            # Can toggle cards
            for i in range(min(len(state.hand), 8)):
                mask[2 + i] = True

        elif state.phase == GamePhase.BLIND_SELECT:
            mask[10] = True  # Small blind
            mask[11] = True  # Big blind
            mask[12] = True  # Boss blind
            mask[13] = True  # Skip

        elif state.phase == GamePhase.SHOP:
            mask[14] = True  # End shop

        return mask

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != "human":
            return

        state = self.game.state
        print(f"\n=== Balatro - Ante {state.ante} ===")
        print(f"Phase: {state.phase.name}")
        print(f"Money: ${state.money}")

        if state.blind:
            print(f"Blind: {state.blind.blind_type} ({state.score}/{state.blind.chips_required})")

        print(f"Hands: {state.hands_remaining}, Discards: {state.discards_remaining}")

        if state.hand:
            print("Hand:", end=" ")
            for i, card in enumerate(state.hand):
                selected = "*" if card in state.played_cards else " "
                print(f"{selected}[{i}]{card.rank.name[0]}{card.suit.name[0]}", end=" ")
            print()

    def close(self) -> None:
        """Clean up resources."""
        pass


def make_env(
    max_ante: int = 8,
    seed: int | None = None,
) -> BalatroEnv:
    """Create a Balatro environment.

    Args:
        max_ante: Maximum ante for curriculum.
        seed: Random seed.

    Returns:
        Configured BalatroEnv instance.
    """
    return BalatroEnv(max_ante=max_ante, seed=seed)
