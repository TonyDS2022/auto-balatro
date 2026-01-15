"""Game constants and enumerations for Auto-Balatro.

This module defines all the core constants, enumerations, and lookup tables
used throughout the Auto-Balatro system. Values are based on the Balatro
game mechanics.
"""

from enum import Enum, auto
from typing import Dict, Any


class Suit(Enum):
    """Card suits in standard playing card deck."""

    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Rank(Enum):
    """Card ranks with numeric values for comparison.

    Values range from 2 (Two) to 14 (Ace) for ordering purposes.
    """

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class HandType(Enum):
    """Poker hand types in Balatro, ordered by base strength.

    Includes standard poker hands plus Balatro-specific hands like
    Five of a Kind, Flush House, and Flush Five.
    """

    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()
    FIVE_OF_A_KIND = auto()
    FLUSH_HOUSE = auto()
    FLUSH_FIVE = auto()


class GamePhase(Enum):
    """Current phase of the game state machine."""

    MAIN_MENU = auto()
    BLIND_SELECT = auto()
    PLAYING = auto()
    SCORING = auto()
    SHOP = auto()
    GAME_OVER = auto()
    PAUSE = auto()


class Enhancement(Enum):
    """Card enhancement types that modify scoring behavior."""

    NONE = auto()
    BONUS = auto()
    MULT = auto()
    WILD = auto()
    GLASS = auto()
    STEEL = auto()
    STONE = auto()
    GOLD = auto()
    LUCKY = auto()


class Edition(Enum):
    """Card edition types that provide additional bonuses."""

    NONE = auto()
    FOIL = auto()
    HOLOGRAPHIC = auto()
    POLYCHROME = auto()
    NEGATIVE = auto()


class ActionType(Enum):
    """Types of actions the agent can take during gameplay."""

    PLAY_HAND = auto()
    DISCARD = auto()
    USE_CONSUMABLE = auto()
    BUY_ITEM = auto()
    SELL_ITEM = auto()
    REROLL_SHOP = auto()
    SELECT_BLIND = auto()
    SKIP_BLIND = auto()


# Base chip values for each hand type
HAND_BASE_CHIPS: Dict[HandType, int] = {
    HandType.HIGH_CARD: 5,
    HandType.PAIR: 10,
    HandType.TWO_PAIR: 20,
    HandType.THREE_OF_A_KIND: 30,
    HandType.STRAIGHT: 30,
    HandType.FLUSH: 35,
    HandType.FULL_HOUSE: 40,
    HandType.FOUR_OF_A_KIND: 60,
    HandType.STRAIGHT_FLUSH: 100,
    HandType.FIVE_OF_A_KIND: 120,
    HandType.FLUSH_HOUSE: 140,
    HandType.FLUSH_FIVE: 160,
}

# Base multiplier values for each hand type
HAND_BASE_MULT: Dict[HandType, int] = {
    HandType.HIGH_CARD: 1,
    HandType.PAIR: 2,
    HandType.TWO_PAIR: 2,
    HandType.THREE_OF_A_KIND: 3,
    HandType.STRAIGHT: 4,
    HandType.FLUSH: 4,
    HandType.FULL_HOUSE: 4,
    HandType.FOUR_OF_A_KIND: 7,
    HandType.STRAIGHT_FLUSH: 8,
    HandType.FIVE_OF_A_KIND: 12,
    HandType.FLUSH_HOUSE: 14,
    HandType.FLUSH_FIVE: 16,
}

# Chip requirements for each ante and blind type
# Structure: ante -> {small_blind, big_blind, boss_blind}
BLIND_REQUIREMENTS: Dict[int, Dict[str, int]] = {
    1: {"small_blind": 300, "big_blind": 450, "boss_blind": 600},
    2: {"small_blind": 800, "big_blind": 1200, "boss_blind": 1600},
    3: {"small_blind": 2000, "big_blind": 3000, "boss_blind": 4000},
    4: {"small_blind": 5000, "big_blind": 7500, "boss_blind": 10000},
    5: {"small_blind": 11000, "big_blind": 16500, "boss_blind": 22000},
    6: {"small_blind": 20000, "big_blind": 30000, "boss_blind": 40000},
    7: {"small_blind": 35000, "big_blind": 52500, "boss_blind": 70000},
    8: {"small_blind": 50000, "big_blind": 75000, "boss_blind": 100000},
}

# Enhancement effects and their bonus values
ENHANCEMENT_EFFECTS: Dict[Enhancement, Dict[str, Any]] = {
    Enhancement.NONE: {},
    Enhancement.BONUS: {
        "chips": 30,
        "description": "Adds +30 chips when scored",
    },
    Enhancement.MULT: {
        "mult": 4,
        "description": "Adds +4 mult when scored",
    },
    Enhancement.WILD: {
        "wild": True,
        "description": "Can be used as any suit",
    },
    Enhancement.GLASS: {
        "x_mult": 2.0,
        "destroy_chance": 0.25,
        "description": "x2 Mult, 1 in 4 chance to destroy after scoring",
    },
    Enhancement.STEEL: {
        "x_mult": 1.5,
        "triggers_in_hand": True,
        "description": "x1.5 Mult while in hand",
    },
    Enhancement.STONE: {
        "chips": 50,
        "no_rank_suit": True,
        "description": "+50 chips, no rank or suit",
    },
    Enhancement.GOLD: {
        "money_on_round_end": 3,
        "description": "$3 at end of round if in hand",
    },
    Enhancement.LUCKY: {
        "mult_chance": 0.2,
        "mult_bonus": 20,
        "money_chance": 0.067,
        "money_bonus": 20,
        "description": "1 in 5 chance for +20 Mult, 1 in 15 chance for $20",
    },
}

# Edition effects and their bonus values
EDITION_EFFECTS: Dict[Edition, Dict[str, Any]] = {
    Edition.NONE: {},
    Edition.FOIL: {
        "chips": 50,
        "description": "+50 chips",
    },
    Edition.HOLOGRAPHIC: {
        "mult": 10,
        "description": "+10 Mult",
    },
    Edition.POLYCHROME: {
        "x_mult": 1.5,
        "description": "x1.5 Mult",
    },
    Edition.NEGATIVE: {
        "joker_slot": 1,
        "description": "+1 Joker slot",
    },
}

# Chip value contribution for each card rank
RANK_CHIPS: Dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 10,
    Rank.QUEEN: 10,
    Rank.KING: 10,
    Rank.ACE: 11,
}

# Maximum cards allowed in a played hand
MAX_HAND_SIZE: int = 5

# Starting values for a new run
STARTING_MONEY: int = 4
STARTING_HANDS: int = 4
STARTING_DISCARDS: int = 3
STARTING_HAND_SIZE: int = 8
STARTING_JOKER_SLOTS: int = 5

# Shop constants
SHOP_REROLL_COST: int = 5
SHOP_SLOTS: int = 2
BOOSTER_PACK_SLOTS: int = 2

# Card sell values
BASE_CARD_SELL_VALUE: int = 1
JOKER_BASE_SELL_DIVISOR: int = 2  # Jokers sell for cost / 2
