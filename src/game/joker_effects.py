"""Joker effects module for Auto-Balatro.

Defines joker types and their effects on hand scoring. Jokers are the primary
way to build powerful scoring combinations in Balatro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from src.game.constants import HandType, Rank, Suit

if TYPE_CHECKING:
    from src.game.hand_evaluator import Card, HandResult

logger = logging.getLogger(__name__)


class JokerRarity(Enum):
    """Joker rarity tiers."""
    COMMON = auto()
    UNCOMMON = auto()
    RARE = auto()
    LEGENDARY = auto()


class JokerType(Enum):
    """Types of joker effects."""
    ADDITIVE_MULT = auto()      # +mult
    MULTIPLICATIVE_MULT = auto()  # xmult
    ADDITIVE_CHIPS = auto()     # +chips
    ECONOMY = auto()            # Money generation
    SCALING = auto()            # Grows over time
    CONDITIONAL = auto()        # Effect based on condition
    RETRIGGER = auto()          # Triggers cards multiple times
    SPECIAL = auto()            # Unique effects


@dataclass
class JokerEffect:
    """Defines the effect a joker has on scoring.

    Attributes:
        chips: Additive chips bonus.
        mult: Additive mult bonus.
        x_mult: Multiplicative mult bonus.
        money: Money generated per trigger.
        condition: Function to check if effect applies.
        description: Human-readable effect description.
    """

    chips: int = 0
    mult: int = 0
    x_mult: float = 1.0
    money: int = 0
    condition: Callable[..., bool] | None = None
    description: str = ""


@dataclass
class Joker:
    """Represents a joker card.

    Attributes:
        name: Joker name.
        joker_id: Unique identifier.
        rarity: Rarity tier.
        joker_type: Type of effect.
        base_effect: Base effect values.
        buy_price: Shop purchase price.
        sell_price: Sell value (usually half of buy).
        edition: Edition (foil, holo, poly, negative).
        scaling_value: Current scaling bonus (for scaling jokers).
    """

    name: str
    joker_id: str
    rarity: JokerRarity = JokerRarity.COMMON
    joker_type: JokerType = JokerType.ADDITIVE_MULT
    base_effect: JokerEffect = field(default_factory=JokerEffect)
    buy_price: int = 4
    sell_price: int = 2
    edition: str | None = None
    scaling_value: int = 0

    def get_effect(
        self,
        hand_result: HandResult | None = None,
        played_cards: list[Card] | None = None,
        held_cards: list[Card] | None = None,
        context: dict | None = None,
    ) -> JokerEffect:
        """Calculate the joker's effect for current context.

        Args:
            hand_result: Result of hand evaluation.
            played_cards: Cards being played.
            held_cards: Cards held in hand.
            context: Additional context (ante, blind, money, etc.).

        Returns:
            JokerEffect with calculated bonuses.
        """
        effect = JokerEffect(
            chips=self.base_effect.chips,
            mult=self.base_effect.mult,
            x_mult=self.base_effect.x_mult,
            money=self.base_effect.money,
        )

        # Check condition if present
        if self.base_effect.condition:
            if not self.base_effect.condition(
                hand_result, played_cards, held_cards, context
            ):
                return JokerEffect()  # No effect if condition not met

        # Add scaling bonus for scaling jokers
        if self.joker_type == JokerType.SCALING:
            effect.mult += self.scaling_value

        # Apply edition bonuses
        if self.edition == "foil":
            effect.chips += 50
        elif self.edition == "holographic":
            effect.mult += 10
        elif self.edition == "polychrome":
            effect.x_mult *= 1.5

        return effect

    def trigger_scaling(self, amount: int = 1) -> None:
        """Trigger scaling increase for scaling jokers.

        Args:
            amount: Amount to increase scaling by.
        """
        if self.joker_type == JokerType.SCALING:
            self.scaling_value += amount


# =============================================================================
# Joker Definitions
# =============================================================================

def _condition_pair_or_better(
    hand_result: HandResult | None, *args
) -> bool:
    """Condition: hand is pair or better."""
    if hand_result is None:
        return False
    return hand_result.hand_type.value >= HandType.PAIR.value


def _condition_contains_suit(suit: Suit):
    """Create condition for containing specific suit."""
    def check(
        hand_result: HandResult | None,
        played_cards: list[Card] | None,
        *args
    ) -> bool:
        if not played_cards:
            return False
        return any(c.suit == suit for c in played_cards)
    return check


def _condition_hand_type(hand_type: HandType):
    """Create condition for specific hand type."""
    def check(hand_result: HandResult | None, *args) -> bool:
        if hand_result is None:
            return False
        return hand_result.hand_type == hand_type
    return check


# Common Jokers
JOKER_DEFINITIONS: dict[str, Joker] = {
    # === Additive Mult Jokers ===
    "joker": Joker(
        name="Joker",
        joker_id="joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.ADDITIVE_MULT,
        base_effect=JokerEffect(mult=4, description="+4 Mult"),
        buy_price=2,
        sell_price=1,
    ),
    "greedy_joker": Joker(
        name="Greedy Joker",
        joker_id="greedy_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.CONDITIONAL,
        base_effect=JokerEffect(
            mult=3,
            condition=_condition_contains_suit(Suit.DIAMONDS),
            description="+3 Mult if played hand contains a Diamond",
        ),
        buy_price=5,
        sell_price=2,
    ),
    "lusty_joker": Joker(
        name="Lusty Joker",
        joker_id="lusty_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.CONDITIONAL,
        base_effect=JokerEffect(
            mult=3,
            condition=_condition_contains_suit(Suit.HEARTS),
            description="+3 Mult if played hand contains a Heart",
        ),
        buy_price=5,
        sell_price=2,
    ),
    "wrathful_joker": Joker(
        name="Wrathful Joker",
        joker_id="wrathful_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.CONDITIONAL,
        base_effect=JokerEffect(
            mult=3,
            condition=_condition_contains_suit(Suit.SPADES),
            description="+3 Mult if played hand contains a Spade",
        ),
        buy_price=5,
        sell_price=2,
    ),
    "gluttonous_joker": Joker(
        name="Gluttonous Joker",
        joker_id="gluttonous_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.CONDITIONAL,
        base_effect=JokerEffect(
            mult=3,
            condition=_condition_contains_suit(Suit.CLUBS),
            description="+3 Mult if played hand contains a Club",
        ),
        buy_price=5,
        sell_price=2,
    ),

    # === Multiplicative Mult Jokers ===
    "the_duo": Joker(
        name="The Duo",
        joker_id="the_duo",
        rarity=JokerRarity.RARE,
        joker_type=JokerType.MULTIPLICATIVE_MULT,
        base_effect=JokerEffect(
            x_mult=2.0,
            condition=_condition_hand_type(HandType.PAIR),
            description="X2 Mult if played hand contains a Pair",
        ),
        buy_price=8,
        sell_price=4,
    ),
    "the_trio": Joker(
        name="The Trio",
        joker_id="the_trio",
        rarity=JokerRarity.RARE,
        joker_type=JokerType.MULTIPLICATIVE_MULT,
        base_effect=JokerEffect(
            x_mult=3.0,
            condition=_condition_hand_type(HandType.THREE_OF_A_KIND),
            description="X3 Mult if played hand contains Three of a Kind",
        ),
        buy_price=8,
        sell_price=4,
    ),
    "the_family": Joker(
        name="The Family",
        joker_id="the_family",
        rarity=JokerRarity.RARE,
        joker_type=JokerType.MULTIPLICATIVE_MULT,
        base_effect=JokerEffect(
            x_mult=4.0,
            condition=_condition_hand_type(HandType.FOUR_OF_A_KIND),
            description="X4 Mult if played hand contains Four of a Kind",
        ),
        buy_price=8,
        sell_price=4,
    ),
    "the_order": Joker(
        name="The Order",
        joker_id="the_order",
        rarity=JokerRarity.RARE,
        joker_type=JokerType.MULTIPLICATIVE_MULT,
        base_effect=JokerEffect(
            x_mult=3.0,
            condition=_condition_hand_type(HandType.STRAIGHT),
            description="X3 Mult if played hand contains a Straight",
        ),
        buy_price=8,
        sell_price=4,
    ),
    "the_tribe": Joker(
        name="The Tribe",
        joker_id="the_tribe",
        rarity=JokerRarity.RARE,
        joker_type=JokerType.MULTIPLICATIVE_MULT,
        base_effect=JokerEffect(
            x_mult=2.0,
            condition=_condition_hand_type(HandType.FLUSH),
            description="X2 Mult if played hand contains a Flush",
        ),
        buy_price=8,
        sell_price=4,
    ),

    # === Chip Jokers ===
    "banner": Joker(
        name="Banner",
        joker_id="banner",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.ADDITIVE_CHIPS,
        base_effect=JokerEffect(chips=30, description="+30 Chips for each discard remaining"),
        buy_price=5,
        sell_price=2,
    ),
    "scary_face": Joker(
        name="Scary Face",
        joker_id="scary_face",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.ADDITIVE_CHIPS,
        base_effect=JokerEffect(chips=30, description="+30 Chips if played hand contains a face card"),
        buy_price=4,
        sell_price=2,
    ),

    # === Scaling Jokers ===
    "ice_cream": Joker(
        name="Ice Cream",
        joker_id="ice_cream",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.SCALING,
        base_effect=JokerEffect(chips=100, description="+100 Chips, -5 Chips per hand played"),
        buy_price=5,
        sell_price=2,
    ),
    "red_card": Joker(
        name="Red Card",
        joker_id="red_card",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.SCALING,
        base_effect=JokerEffect(mult=3, description="+3 Mult, gains +3 Mult when blind is skipped"),
        buy_price=5,
        sell_price=2,
    ),
    "green_joker": Joker(
        name="Green Joker",
        joker_id="green_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.SCALING,
        base_effect=JokerEffect(mult=1, description="+1 Mult per hand played, resets at end of round"),
        buy_price=5,
        sell_price=2,
    ),

    # === Economy Jokers ===
    "golden_joker": Joker(
        name="Golden Joker",
        joker_id="golden_joker",
        rarity=JokerRarity.COMMON,
        joker_type=JokerType.ECONOMY,
        base_effect=JokerEffect(money=4, description="Earn $4 at end of round"),
        buy_price=6,
        sell_price=3,
    ),
    "rocket": Joker(
        name="Rocket",
        joker_id="rocket",
        rarity=JokerRarity.UNCOMMON,
        joker_type=JokerType.ECONOMY,
        base_effect=JokerEffect(money=1, description="Earn $1 at end of round, payout increases by $2 when Boss Blind is defeated"),
        buy_price=6,
        sell_price=3,
    ),
}


class JokerManager:
    """Manages jokers and calculates their combined effects.

    Attributes:
        jokers: List of active jokers.
        max_jokers: Maximum joker slots.
    """

    def __init__(self, max_jokers: int = 5) -> None:
        """Initialize joker manager.

        Args:
            max_jokers: Maximum number of joker slots.
        """
        self.jokers: list[Joker] = []
        self.max_jokers = max_jokers
        logger.info(f"JokerManager initialized with {max_jokers} slots")

    def add_joker(self, joker: Joker) -> bool:
        """Add a joker if slots available.

        Args:
            joker: Joker to add.

        Returns:
            True if added, False if no slots.
        """
        # Negative edition jokers don't take a slot
        effective_count = sum(1 for j in self.jokers if j.edition != "negative")

        if joker.edition == "negative" or effective_count < self.max_jokers:
            self.jokers.append(joker)
            logger.debug(f"Added joker: {joker.name}")
            return True

        logger.debug(f"Cannot add joker {joker.name}: no slots available")
        return False

    def remove_joker(self, joker_id: str) -> Joker | None:
        """Remove a joker by ID.

        Args:
            joker_id: ID of joker to remove.

        Returns:
            Removed joker or None if not found.
        """
        for i, joker in enumerate(self.jokers):
            if joker.joker_id == joker_id:
                return self.jokers.pop(i)
        return None

    def sell_joker(self, joker_id: str) -> int:
        """Sell a joker and return sell price.

        Args:
            joker_id: ID of joker to sell.

        Returns:
            Sell price (0 if joker not found).
        """
        joker = self.remove_joker(joker_id)
        if joker:
            logger.debug(f"Sold joker {joker.name} for ${joker.sell_price}")
            return joker.sell_price
        return 0

    def calculate_effects(
        self,
        hand_result: HandResult,
        played_cards: list[Card],
        held_cards: list[Card] | None = None,
        context: dict | None = None,
    ) -> tuple[int, int, float, int]:
        """Calculate combined effects of all jokers.

        Args:
            hand_result: Result of hand evaluation.
            played_cards: Cards being played.
            held_cards: Cards held in hand.
            context: Additional context.

        Returns:
            Tuple of (total_chips, total_mult, total_x_mult, total_money).
        """
        total_chips = 0
        total_mult = 0
        total_x_mult = 1.0
        total_money = 0

        for joker in self.jokers:
            effect = joker.get_effect(
                hand_result, played_cards, held_cards, context
            )
            total_chips += effect.chips
            total_mult += effect.mult
            total_x_mult *= effect.x_mult
            total_money += effect.money

        return total_chips, total_mult, total_x_mult, total_money

    def apply_to_score(
        self,
        hand_result: HandResult,
        played_cards: list[Card],
        held_cards: list[Card] | None = None,
        context: dict | None = None,
    ) -> int:
        """Apply joker effects to a hand result and return final score.

        Args:
            hand_result: Result of hand evaluation.
            played_cards: Cards being played.
            held_cards: Cards held in hand.
            context: Additional context.

        Returns:
            Final score after joker effects.
        """
        chips, mult, x_mult, _ = self.calculate_effects(
            hand_result, played_cards, held_cards, context
        )

        # Apply joker bonuses to hand result
        total_chips = hand_result.base_chips + hand_result.card_chips + chips
        total_mult = (hand_result.base_mult + hand_result.mult_bonus + mult)
        total_mult *= hand_result.mult_multiplier * x_mult

        return int(total_chips * total_mult)

    def trigger_end_of_round(self) -> int:
        """Trigger end-of-round effects and return money earned.

        Returns:
            Total money earned from joker effects.
        """
        money = 0
        for joker in self.jokers:
            if joker.joker_type == JokerType.ECONOMY:
                effect = joker.get_effect()
                money += effect.money
        return money

    def get_joker(self, joker_id: str) -> Joker | None:
        """Get a joker by ID.

        Args:
            joker_id: ID of joker to find.

        Returns:
            Joker or None if not found.
        """
        for joker in self.jokers:
            if joker.joker_id == joker_id:
                return joker
        return None


def create_joker(joker_id: str, edition: str | None = None) -> Joker | None:
    """Create a joker instance from definitions.

    Args:
        joker_id: ID of joker to create.
        edition: Optional edition (foil, holographic, polychrome, negative).

    Returns:
        New Joker instance or None if not found.
    """
    if joker_id not in JOKER_DEFINITIONS:
        logger.warning(f"Unknown joker ID: {joker_id}")
        return None

    # Create a copy with optional edition
    template = JOKER_DEFINITIONS[joker_id]
    joker = Joker(
        name=template.name,
        joker_id=template.joker_id,
        rarity=template.rarity,
        joker_type=template.joker_type,
        base_effect=template.base_effect,
        buy_price=template.buy_price,
        sell_price=template.sell_price,
        edition=edition,
        scaling_value=template.scaling_value,
    )
    return joker
