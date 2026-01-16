"""Game state machine for Auto-Balatro.

Manages the complete game state including deck, hand, jokers, resources,
and transitions between game phases.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.game.constants import (
    ActionType,
    GamePhase,
    Rank,
    Suit,
    BLIND_REQUIREMENTS,
    STARTING_MONEY,
    STARTING_HANDS,
    STARTING_DISCARDS,
    MAX_HAND_SIZE,
)
from src.game.hand_evaluator import Card, HandEvaluator, HandResult
from src.game.joker_effects import Joker, JokerManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class BlindInfo:
    """Information about current blind.

    Attributes:
        blind_type: Type of blind (small, big, boss).
        chips_required: Chips needed to beat the blind.
        reward: Money reward for beating the blind.
        boss_effect: Special boss blind effect (if any).
    """

    blind_type: str  # "small", "big", "boss"
    chips_required: int
    reward: int = 3
    boss_effect: str | None = None


@dataclass
class GameState:
    """Complete game state.

    Attributes:
        phase: Current game phase.
        ante: Current ante (1-8).
        blind: Current blind info.
        score: Current round score (chips earned).
        money: Player's money.
        hands_remaining: Hands left to play this round.
        discards_remaining: Discards left this round.
        deck: Cards remaining in deck.
        hand: Cards in player's hand.
        played_cards: Cards currently selected/played.
        jokers: JokerManager for active jokers.
        consumables: Active consumable items.
        round_hands_played: Hands played this round (for scoring).
    """

    phase: GamePhase = GamePhase.MAIN_MENU
    ante: int = 1
    blind: BlindInfo | None = None
    score: int = 0
    money: int = STARTING_MONEY
    hands_remaining: int = STARTING_HANDS
    discards_remaining: int = STARTING_DISCARDS
    deck: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    played_cards: list[Card] = field(default_factory=list)
    jokers: JokerManager = field(default_factory=JokerManager)
    consumables: list = field(default_factory=list)
    round_hands_played: int = 0
    max_hand_size: int = MAX_HAND_SIZE

    def copy(self) -> GameState:
        """Create a shallow copy of the game state."""
        return GameState(
            phase=self.phase,
            ante=self.ante,
            blind=self.blind,
            score=self.score,
            money=self.money,
            hands_remaining=self.hands_remaining,
            discards_remaining=self.discards_remaining,
            deck=list(self.deck),
            hand=list(self.hand),
            played_cards=list(self.played_cards),
            jokers=self.jokers,  # Shared reference
            consumables=list(self.consumables),
            round_hands_played=self.round_hands_played,
            max_hand_size=self.max_hand_size,
        )


class GameStateMachine:
    """Manages game state transitions and logic.

    Handles all game actions, phase transitions, and scoring calculations.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the game state machine.

        Args:
            seed: Random seed for reproducibility.
        """
        self.state = GameState()
        self.evaluator = HandEvaluator()
        self.rng = random.Random(seed)

        logger.info(f"GameStateMachine initialized with seed={seed}")

    def new_game(self) -> GameState:
        """Start a new game.

        Returns:
            Initial game state.
        """
        self.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            money=STARTING_MONEY,
            hands_remaining=STARTING_HANDS,
            discards_remaining=STARTING_DISCARDS,
            jokers=JokerManager(),
        )

        # Create and shuffle standard deck
        self._create_deck()

        logger.info("New game started")
        return self.state

    def _create_deck(self) -> None:
        """Create a standard 52-card deck."""
        self.state.deck = []
        for suit in Suit:
            for rank in Rank:
                self.state.deck.append(Card(rank=rank, suit=suit))
        self.rng.shuffle(self.state.deck)

    def select_blind(self, blind_type: str) -> GameState:
        """Select a blind to play.

        Args:
            blind_type: "small", "big", or "boss".

        Returns:
            Updated game state.
        """
        if self.state.phase != GamePhase.BLIND_SELECT:
            logger.warning(f"Cannot select blind in phase {self.state.phase}")
            return self.state

        ante_reqs = BLIND_REQUIREMENTS.get(self.state.ante, {})
        chips_key = f"{blind_type}_blind"
        chips_required = ante_reqs.get(chips_key, 300)

        # Rewards
        rewards = {"small": 3, "big": 4, "boss": 5}

        self.state.blind = BlindInfo(
            blind_type=blind_type,
            chips_required=chips_required,
            reward=rewards.get(blind_type, 3),
        )

        # Reset for new round
        self.state.score = 0
        self.state.hands_remaining = STARTING_HANDS
        self.state.discards_remaining = STARTING_DISCARDS
        self.state.round_hands_played = 0

        # Shuffle deck and deal hand
        self.rng.shuffle(self.state.deck)
        self._deal_hand()

        self.state.phase = GamePhase.PLAYING
        logger.info(f"Selected {blind_type} blind, need {chips_required} chips")

        return self.state

    def skip_blind(self) -> GameState:
        """Skip the current blind (small/big only).

        Returns:
            Updated game state.
        """
        if self.state.phase != GamePhase.BLIND_SELECT:
            return self.state

        # Can only skip small/big blinds
        # Move to next blind or shop
        logger.info("Blind skipped")

        # Trigger scaling jokers that benefit from skipping
        for joker in self.state.jokers.jokers:
            if joker.joker_id == "red_card":
                joker.trigger_scaling(3)

        return self.state

    def _deal_hand(self) -> None:
        """Deal cards to fill hand to max size."""
        while len(self.state.hand) < self.state.max_hand_size and self.state.deck:
            self.state.hand.append(self.state.deck.pop())

    def select_cards(self, indices: list[int]) -> GameState:
        """Select cards from hand for playing/discarding.

        Args:
            indices: Indices of cards in hand to select.

        Returns:
            Updated game state.
        """
        if self.state.phase != GamePhase.PLAYING:
            return self.state

        self.state.played_cards = []
        for i in indices:
            if 0 <= i < len(self.state.hand):
                self.state.played_cards.append(self.state.hand[i])

        return self.state

    def play_hand(self) -> tuple[GameState, HandResult | None]:
        """Play the selected cards.

        Returns:
            Tuple of (updated state, hand result).
        """
        if self.state.phase != GamePhase.PLAYING:
            return self.state, None

        if not self.state.played_cards:
            logger.warning("No cards selected to play")
            return self.state, None

        if self.state.hands_remaining <= 0:
            logger.warning("No hands remaining")
            return self.state, None

        # Evaluate hand
        hand_result = self.evaluator.evaluate(self.state.played_cards)

        # Apply joker effects
        held_cards = [c for c in self.state.hand if c not in self.state.played_cards]
        final_score = self.state.jokers.apply_to_score(
            hand_result,
            self.state.played_cards,
            held_cards,
            {"ante": self.state.ante, "money": self.state.money},
        )

        # Update state
        self.state.score += final_score
        self.state.hands_remaining -= 1
        self.state.round_hands_played += 1

        # Remove played cards from hand
        for card in self.state.played_cards:
            if card in self.state.hand:
                self.state.hand.remove(card)

        self.state.played_cards = []

        # Deal new cards
        self._deal_hand()

        logger.info(
            f"Played {hand_result.hand_type.name} for {final_score} chips "
            f"(total: {self.state.score}/{self.state.blind.chips_required if self.state.blind else 0})"
        )

        # Check if blind is beaten
        if self.state.blind and self.state.score >= self.state.blind.chips_required:
            self._complete_blind()

        # Check if out of hands
        elif self.state.hands_remaining <= 0:
            self._check_game_over()

        return self.state, hand_result

    def discard(self) -> GameState:
        """Discard selected cards and draw new ones.

        Returns:
            Updated game state.
        """
        if self.state.phase != GamePhase.PLAYING:
            return self.state

        if not self.state.played_cards:
            logger.warning("No cards selected to discard")
            return self.state

        if self.state.discards_remaining <= 0:
            logger.warning("No discards remaining")
            return self.state

        # Remove discarded cards
        for card in self.state.played_cards:
            if card in self.state.hand:
                self.state.hand.remove(card)
                # Add to bottom of deck
                self.state.deck.insert(0, card)

        self.state.played_cards = []
        self.state.discards_remaining -= 1

        # Deal new cards
        self._deal_hand()

        logger.debug(f"Discarded cards, {self.state.discards_remaining} discards remaining")

        return self.state

    def _complete_blind(self) -> None:
        """Handle blind completion."""
        if not self.state.blind:
            return

        # Award money
        self.state.money += self.state.blind.reward

        # Interest (1$ per $5, max $5)
        interest = min(self.state.money // 5, 5)
        self.state.money += interest

        # Joker end-of-round money
        joker_money = self.state.jokers.trigger_end_of_round()
        self.state.money += joker_money

        logger.info(
            f"Blind beaten! Reward: ${self.state.blind.reward}, "
            f"Interest: ${interest}, Jokers: ${joker_money}"
        )

        # Move to shop or next blind
        if self.state.blind.blind_type == "boss":
            # Next ante
            self.state.ante += 1
            if self.state.ante > 8:
                self._win_game()
            else:
                self.state.phase = GamePhase.SHOP
        else:
            # Next blind in ante
            self.state.phase = GamePhase.SHOP

        self.state.blind = None

    def _check_game_over(self) -> None:
        """Check if game is over (failed to beat blind)."""
        if self.state.blind and self.state.score < self.state.blind.chips_required:
            self.state.phase = GamePhase.GAME_OVER
            logger.info("Game Over - failed to beat blind")

    def _win_game(self) -> None:
        """Handle game win."""
        self.state.phase = GamePhase.GAME_OVER
        logger.info("Game Won! Completed all 8 antes")

    def end_shop(self) -> GameState:
        """End shopping phase and move to blind select.

        Returns:
            Updated game state.
        """
        if self.state.phase != GamePhase.SHOP:
            return self.state

        self.state.phase = GamePhase.BLIND_SELECT
        return self.state

    def buy_item(self, item_type: str, item_id: str, cost: int) -> bool:
        """Buy an item from the shop.

        Args:
            item_type: Type of item ("joker", "consumable", "pack").
            item_id: ID of item to buy.
            cost: Cost of the item.

        Returns:
            True if purchase successful.
        """
        if self.state.money < cost:
            return False

        self.state.money -= cost
        logger.debug(f"Bought {item_type} {item_id} for ${cost}")
        return True

    def sell_joker(self, joker_id: str) -> int:
        """Sell a joker.

        Args:
            joker_id: ID of joker to sell.

        Returns:
            Money received from sale.
        """
        money = self.state.jokers.sell_joker(joker_id)
        self.state.money += money
        return money

    def get_valid_actions(self) -> list[ActionType]:
        """Get list of valid actions in current state.

        Returns:
            List of valid ActionType values.
        """
        actions = []

        if self.state.phase == GamePhase.BLIND_SELECT:
            actions.extend([ActionType.SELECT_BLIND, ActionType.SKIP_BLIND])

        elif self.state.phase == GamePhase.PLAYING:
            if self.state.hands_remaining > 0:
                actions.append(ActionType.PLAY_HAND)
            if self.state.discards_remaining > 0:
                actions.append(ActionType.DISCARD)
            if self.state.consumables:
                actions.append(ActionType.USE_CONSUMABLE)

        elif self.state.phase == GamePhase.SHOP:
            actions.extend([
                ActionType.BUY_ITEM,
                ActionType.SELL_ITEM,
                ActionType.REROLL_SHOP,
            ])

        return actions

    def get_state_dict(self) -> dict:
        """Get state as a dictionary for observation.

        Returns:
            Dictionary representation of game state.
        """
        return {
            "phase": self.state.phase.value,
            "ante": self.state.ante,
            "score": self.state.score,
            "chips_needed": self.state.blind.chips_required if self.state.blind else 0,
            "money": self.state.money,
            "hands_remaining": self.state.hands_remaining,
            "discards_remaining": self.state.discards_remaining,
            "hand_size": len(self.state.hand),
            "deck_size": len(self.state.deck),
            "joker_count": len(self.state.jokers.jokers),
            "round_hands_played": self.state.round_hands_played,
        }
