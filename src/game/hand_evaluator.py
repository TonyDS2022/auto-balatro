"""Poker hand evaluation and Balatro scoring module.

Evaluates poker hands and calculates scores using Balatro's scoring formula:
Final Score = (Base Chips + Card Chips + Bonus Chips) × (Base Mult × Mult Bonuses)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.game.constants import (
    Enhancement,
    Edition,
    HandType,
    Rank,
    Suit,
    HAND_BASE_CHIPS,
    HAND_BASE_MULT,
    RANK_CHIPS,
    ENHANCEMENT_EFFECTS,
    EDITION_EFFECTS,
)

if TYPE_CHECKING:
    from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class Card:
    """Represents a playing card with enhancements.

    Attributes:
        rank: Card rank (TWO through ACE).
        suit: Card suit (HEARTS, DIAMONDS, CLUBS, SPADES).
        enhancement: Card enhancement (BONUS, MULT, WILD, etc.).
        edition: Card edition (FOIL, HOLOGRAPHIC, POLYCHROME, etc.).
        seal: Card seal type (if any).
    """

    rank: Rank
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: str | None = None

    @property
    def chips(self) -> int:
        """Get chip value of this card."""
        base = RANK_CHIPS.get(self.rank, 0)
        # Add enhancement bonus chips
        if self.enhancement == Enhancement.BONUS:
            base += ENHANCEMENT_EFFECTS[Enhancement.BONUS].get("chips", 0)
        elif self.enhancement == Enhancement.STONE:
            base += ENHANCEMENT_EFFECTS[Enhancement.STONE].get("chips", 0)
        # Add edition bonus chips
        if self.edition == Edition.FOIL:
            base += EDITION_EFFECTS[Edition.FOIL].get("chips", 0)
        return base

    def __repr__(self) -> str:
        s = f"{self.rank.name} of {self.suit.name}"
        if self.enhancement != Enhancement.NONE:
            s += f" [{self.enhancement.name}]"
        if self.edition != Edition.NONE:
            s += f" ({self.edition.name})"
        return s


@dataclass
class HandResult:
    """Result of hand evaluation.

    Attributes:
        hand_type: Type of poker hand detected.
        scoring_cards: Cards that contribute to the hand.
        base_chips: Base chips from hand type.
        base_mult: Base multiplier from hand type.
        card_chips: Total chips from scoring cards.
        bonus_chips: Additional chips from enhancements/editions.
        mult_bonus: Additive multiplier bonuses.
        mult_multiplier: Multiplicative multiplier bonuses (x2, x1.5, etc.).
        total_score: Final calculated score.
    """

    hand_type: HandType
    scoring_cards: list[Card] = field(default_factory=list)
    base_chips: int = 0
    base_mult: int = 0
    card_chips: int = 0
    bonus_chips: int = 0
    mult_bonus: int = 0
    mult_multiplier: float = 1.0
    total_score: int = 0

    def calculate_score(self) -> int:
        """Calculate and return total score."""
        total_chips = self.base_chips + self.card_chips + self.bonus_chips
        total_mult = (self.base_mult + self.mult_bonus) * self.mult_multiplier
        self.total_score = int(total_chips * total_mult)
        return self.total_score


class HandEvaluator:
    """Evaluates poker hands and calculates Balatro scores.

    Supports all standard poker hands plus Balatro-specific hands
    (Five of a Kind, Flush House, Flush Five).
    """

    def __init__(self) -> None:
        """Initialize the hand evaluator."""
        logger.info("HandEvaluator initialized")

    def evaluate(self, cards: Sequence[Card]) -> HandResult:
        """Evaluate a hand of cards and return the best hand result.

        Args:
            cards: Sequence of cards to evaluate (typically 5).

        Returns:
            HandResult with detected hand type and scoring breakdown.
        """
        if not cards:
            return HandResult(hand_type=HandType.HIGH_CARD)

        cards_list = list(cards)

        # Check hands from best to worst
        checkers = [
            (HandType.FLUSH_FIVE, self._is_flush_five),
            (HandType.FLUSH_HOUSE, self._is_flush_house),
            (HandType.FIVE_OF_A_KIND, self._is_five_of_a_kind),
            (HandType.STRAIGHT_FLUSH, self._is_straight_flush),
            (HandType.FOUR_OF_A_KIND, self._is_four_of_a_kind),
            (HandType.FULL_HOUSE, self._is_full_house),
            (HandType.FLUSH, self._is_flush),
            (HandType.STRAIGHT, self._is_straight),
            (HandType.THREE_OF_A_KIND, self._is_three_of_a_kind),
            (HandType.TWO_PAIR, self._is_two_pair),
            (HandType.PAIR, self._is_pair),
        ]

        for hand_type, checker in checkers:
            scoring_cards = checker(cards_list)
            if scoring_cards:
                return self._build_result(hand_type, scoring_cards, cards_list)

        # High card - just the highest card
        scoring_cards = [max(cards_list, key=lambda c: c.rank.value)]
        return self._build_result(HandType.HIGH_CARD, scoring_cards, cards_list)

    def _build_result(
        self,
        hand_type: HandType,
        scoring_cards: list[Card],
        all_cards: list[Card],
    ) -> HandResult:
        """Build a HandResult with scoring breakdown.

        Args:
            hand_type: Detected hand type.
            scoring_cards: Cards contributing to the hand.
            all_cards: All cards in hand (for held card bonuses).

        Returns:
            Populated HandResult.
        """
        result = HandResult(
            hand_type=hand_type,
            scoring_cards=scoring_cards,
            base_chips=HAND_BASE_CHIPS.get(hand_type, 0),
            base_mult=HAND_BASE_MULT.get(hand_type, 0),
        )

        # Calculate card chips from scoring cards
        result.card_chips = sum(card.chips for card in scoring_cards)

        # Calculate enhancement bonuses
        for card in scoring_cards:
            if card.enhancement == Enhancement.MULT:
                result.mult_bonus += ENHANCEMENT_EFFECTS[Enhancement.MULT].get("mult", 0)
            elif card.enhancement == Enhancement.GLASS:
                result.mult_multiplier *= ENHANCEMENT_EFFECTS[Enhancement.GLASS].get("x_mult", 1.0)

            # Edition bonuses
            if card.edition == Edition.HOLOGRAPHIC:
                result.mult_bonus += EDITION_EFFECTS[Edition.HOLOGRAPHIC].get("mult", 0)
            elif card.edition == Edition.POLYCHROME:
                result.mult_multiplier *= EDITION_EFFECTS[Edition.POLYCHROME].get("x_mult", 1.0)

        # Steel cards bonus (while held, not played)
        for card in all_cards:
            if card not in scoring_cards and card.enhancement == Enhancement.STEEL:
                result.mult_multiplier *= ENHANCEMENT_EFFECTS[Enhancement.STEEL].get("x_mult", 1.0)

        result.calculate_score()
        return result

    def _get_suits(self, cards: list[Card]) -> list[Suit]:
        """Get suits considering wild cards."""
        suits = []
        for card in cards:
            if card.enhancement == Enhancement.WILD:
                # Wild can be any suit - return all for checking
                suits.append(card.suit)  # Use original for simplicity
            else:
                suits.append(card.suit)
        return suits

    def _is_flush(self, cards: list[Card]) -> list[Card] | None:
        """Check for flush (5 cards same suit)."""
        if len(cards) < 5:
            return None

        suit_counts: Counter[Suit] = Counter()
        for card in cards:
            if card.enhancement == Enhancement.WILD:
                # Wild counts for all suits
                for suit in Suit:
                    suit_counts[suit] += 1
            else:
                suit_counts[card.suit] += 1

        for suit, count in suit_counts.items():
            if count >= 5:
                # Return cards of that suit (+ wilds)
                flush_cards = [
                    c for c in cards
                    if c.suit == suit or c.enhancement == Enhancement.WILD
                ][:5]
                return flush_cards
        return None

    def _is_straight(self, cards: list[Card]) -> list[Card] | None:
        """Check for straight (5 sequential ranks)."""
        if len(cards) < 5:
            return None

        ranks = sorted(set(c.rank.value for c in cards), reverse=True)

        # Check for ace-low straight (A-2-3-4-5)
        if 14 in ranks:  # Ace
            ranks_with_low_ace = sorted(set(
                [1 if r == 14 else r for r in ranks]
            ), reverse=True)
            if self._has_sequential(ranks_with_low_ace, 5):
                return self._get_straight_cards(cards, use_low_ace=True)

        if self._has_sequential(ranks, 5):
            return self._get_straight_cards(cards, use_low_ace=False)

        return None

    def _has_sequential(self, ranks: list[int], length: int) -> bool:
        """Check if ranks contain a sequence of given length."""
        if len(ranks) < length:
            return False

        for i in range(len(ranks) - length + 1):
            is_seq = True
            for j in range(length - 1):
                if ranks[i + j] - ranks[i + j + 1] != 1:
                    is_seq = False
                    break
            if is_seq:
                return True
        return False

    def _get_straight_cards(self, cards: list[Card], use_low_ace: bool) -> list[Card]:
        """Get the cards forming a straight."""
        if use_low_ace:
            sorted_cards = sorted(
                cards,
                key=lambda c: 1 if c.rank == Rank.ACE else c.rank.value,
                reverse=True
            )
        else:
            sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)

        # Take first 5 unique ranks
        seen_ranks: set[int] = set()
        result: list[Card] = []
        for card in sorted_cards:
            rank_val = 1 if (use_low_ace and card.rank == Rank.ACE) else card.rank.value
            if rank_val not in seen_ranks:
                seen_ranks.add(rank_val)
                result.append(card)
            if len(result) == 5:
                break
        return result

    def _is_straight_flush(self, cards: list[Card]) -> list[Card] | None:
        """Check for straight flush."""
        flush_cards = self._is_flush(cards)
        if flush_cards:
            straight_cards = self._is_straight(flush_cards)
            if straight_cards:
                return straight_cards
        return None

    def _get_rank_groups(self, cards: list[Card]) -> dict[Rank, list[Card]]:
        """Group cards by rank."""
        groups: dict[Rank, list[Card]] = {}
        for card in cards:
            if card.rank not in groups:
                groups[card.rank] = []
            groups[card.rank].append(card)
        return groups

    def _is_five_of_a_kind(self, cards: list[Card]) -> list[Card] | None:
        """Check for five of a kind (requires wild cards or special jokers)."""
        groups = self._get_rank_groups(cards)
        for rank, group in groups.items():
            if len(group) >= 5:
                return group[:5]
        return None

    def _is_four_of_a_kind(self, cards: list[Card]) -> list[Card] | None:
        """Check for four of a kind."""
        groups = self._get_rank_groups(cards)
        for rank, group in groups.items():
            if len(group) >= 4:
                return group[:4]
        return None

    def _is_full_house(self, cards: list[Card]) -> list[Card] | None:
        """Check for full house (three of a kind + pair)."""
        groups = self._get_rank_groups(cards)
        three = None
        pair = None

        # Find best three of a kind
        for rank, group in sorted(groups.items(), key=lambda x: x[0].value, reverse=True):
            if len(group) >= 3 and three is None:
                three = group[:3]
            elif len(group) >= 2 and pair is None:
                pair = group[:2]

        if three and pair:
            return three + pair
        return None

    def _is_flush_house(self, cards: list[Card]) -> list[Card] | None:
        """Check for flush house (full house + flush)."""
        full_house = self._is_full_house(cards)
        if full_house:
            if self._is_flush(full_house):
                return full_house
        return None

    def _is_flush_five(self, cards: list[Card]) -> list[Card] | None:
        """Check for flush five (five of a kind + flush)."""
        five_kind = self._is_five_of_a_kind(cards)
        if five_kind:
            if self._is_flush(five_kind):
                return five_kind
        return None

    def _is_three_of_a_kind(self, cards: list[Card]) -> list[Card] | None:
        """Check for three of a kind."""
        groups = self._get_rank_groups(cards)
        for rank, group in sorted(groups.items(), key=lambda x: x[0].value, reverse=True):
            if len(group) >= 3:
                return group[:3]
        return None

    def _is_two_pair(self, cards: list[Card]) -> list[Card] | None:
        """Check for two pair."""
        groups = self._get_rank_groups(cards)
        pairs: list[list[Card]] = []

        for rank, group in sorted(groups.items(), key=lambda x: x[0].value, reverse=True):
            if len(group) >= 2:
                pairs.append(group[:2])
            if len(pairs) == 2:
                return pairs[0] + pairs[1]
        return None

    def _is_pair(self, cards: list[Card]) -> list[Card] | None:
        """Check for pair."""
        groups = self._get_rank_groups(cards)
        for rank, group in sorted(groups.items(), key=lambda x: x[0].value, reverse=True):
            if len(group) >= 2:
                return group[:2]
        return None

    def get_best_hand(
        self,
        hand: Sequence[Card],
        select_count: int = 5,
    ) -> tuple[list[Card], HandResult]:
        """Find the best possible hand from available cards.

        Args:
            hand: Available cards to choose from.
            select_count: Number of cards to select (default 5).

        Returns:
            Tuple of (selected cards, HandResult).
        """
        from itertools import combinations

        if len(hand) <= select_count:
            result = self.evaluate(hand)
            return list(hand), result

        best_cards: list[Card] = []
        best_result = HandResult(hand_type=HandType.HIGH_CARD)

        for combo in combinations(hand, select_count):
            result = self.evaluate(combo)
            if (result.hand_type.value > best_result.hand_type.value or
                (result.hand_type == best_result.hand_type and
                 result.total_score > best_result.total_score)):
                best_cards = list(combo)
                best_result = result

        return best_cards, best_result
