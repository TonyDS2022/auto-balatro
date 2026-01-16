"""Card detection module for Auto-Balatro.

Detects playing cards in game screenshots by finding card contours
and identifying rank/suit through template matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.game.constants import Rank, Suit
from src.vision.template_matcher import TemplateMatcher

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Template name mappings
RANK_TEMPLATES = {
    Rank.TWO: "ranks/2",
    Rank.THREE: "ranks/3",
    Rank.FOUR: "ranks/4",
    Rank.FIVE: "ranks/5",
    Rank.SIX: "ranks/6",
    Rank.SEVEN: "ranks/7",
    Rank.EIGHT: "ranks/8",
    Rank.NINE: "ranks/9",
    Rank.TEN: "ranks/10",
    Rank.JACK: "ranks/J",
    Rank.QUEEN: "ranks/Q",
    Rank.KING: "ranks/K",
    Rank.ACE: "ranks/A",
}

SUIT_TEMPLATES = {
    Suit.HEARTS: "suits/hearts",
    Suit.DIAMONDS: "suits/diamonds",
    Suit.CLUBS: "suits/clubs",
    Suit.SPADES: "suits/spades",
}


@dataclass
class DetectedCard:
    """Represents a detected playing card.

    Attributes:
        rank: Card rank (TWO through ACE).
        suit: Card suit (HEARTS, DIAMONDS, CLUBS, SPADES).
        x: Card center x coordinate.
        y: Card center y coordinate.
        width: Card width in pixels.
        height: Card height in pixels.
        confidence: Detection confidence (0-1).
        selected: Whether card appears selected/highlighted.
    """

    rank: Rank
    suit: Suit
    x: int
    y: int
    width: int
    height: int
    confidence: float
    selected: bool = False

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x1, y1, x2, y2)."""
        half_w = self.width // 2
        half_h = self.height // 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )

    def __repr__(self) -> str:
        return f"{self.rank.name} of {self.suit.name} @ ({self.x}, {self.y})"


class CardDetector:
    """Detects playing cards in game screenshots.

    Uses contour detection to find card shapes, then template matching
    to identify rank and suit of each card.

    Attributes:
        template_matcher: TemplateMatcher instance for rank/suit matching.
        card_aspect_ratio: Expected aspect ratio range for cards (width/height).
        min_card_area: Minimum contour area to consider as a card.
    """

    def __init__(
        self,
        template_matcher: TemplateMatcher | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize the card detector.

        Args:
            template_matcher: TemplateMatcher to use. If None, creates one.
            config: Configuration dictionary with optional keys:
                - rank_region: (x, y, w, h) relative region for rank
                - suit_region: (x, y, w, h) relative region for suit
                - aspect_ratio_min: Minimum card aspect ratio (default: 0.5)
                - aspect_ratio_max: Maximum card aspect ratio (default: 0.9)
                - min_card_area: Minimum card area in pixels (default: 1000)
                - selection_threshold: Y-offset for selected cards (default: 20)
        """
        config = config or {}

        self.template_matcher = template_matcher or TemplateMatcher()

        # Card detection parameters
        self.aspect_ratio_min = config.get("aspect_ratio_min", 0.5)
        self.aspect_ratio_max = config.get("aspect_ratio_max", 0.9)
        self.min_card_area = config.get("min_card_area", 1000)
        self.selection_threshold = config.get("selection_threshold", 20)

        # Rank/suit region as fraction of card size
        # Default: rank in upper-left, suit below rank
        self.rank_region = config.get("rank_region", (0, 0, 0.33, 0.33))
        self.suit_region = config.get("suit_region", (0, 0.15, 0.33, 0.35))

        logger.info("CardDetector initialized")

    def detect_cards(self, image: NDArray[np.uint8]) -> list[DetectedCard]:
        """Detect all cards in an image.

        Args:
            image: BGR image to search for cards.

        Returns:
            List of DetectedCard objects sorted left-to-right.
        """
        # Find card contours
        contours = self._find_card_contours(image)

        if not contours:
            return []

        # Calculate average Y position for selection detection
        avg_y = np.mean([cv2.boundingRect(c)[1] for c in contours])

        cards: list[DetectedCard] = []

        for contour in contours:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)

            # Extract card region
            card_img = image[y : y + h, x : x + w]

            # Identify rank and suit
            rank_result = self._identify_rank(card_img)
            suit_result = self._identify_suit(card_img)

            if rank_result is None or suit_result is None:
                continue

            rank, rank_conf = rank_result
            suit, suit_conf = suit_result

            # Check if card is selected (raised higher than average)
            selected = self._is_card_selected(y, avg_y)

            # Combined confidence
            confidence = (rank_conf + suit_conf) / 2

            cards.append(
                DetectedCard(
                    rank=rank,
                    suit=suit,
                    x=x + w // 2,
                    y=y + h // 2,
                    width=w,
                    height=h,
                    confidence=confidence,
                    selected=selected,
                )
            )

        # Sort left-to-right
        cards.sort(key=lambda c: c.x)

        return cards

    def _find_card_contours(self, image: NDArray[np.uint8]) -> list[NDArray]:
        """Find card-shaped contours in an image.

        Args:
            image: BGR image to search.

        Returns:
            List of contours that match card shape criteria.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        card_contours: list[NDArray] = []

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by area
            area = w * h
            if area < self.min_card_area:
                continue

            # Filter by aspect ratio (width/height for portrait cards)
            aspect = w / h if h > 0 else 0
            if not (self.aspect_ratio_min <= aspect <= self.aspect_ratio_max):
                continue

            # Optional: verify it's approximately rectangular
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) >= 4 and len(approx) <= 8:
                card_contours.append(contour)

        return card_contours

    def _extract_rank_region(self, card_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract the rank region from a card image.

        Args:
            card_image: Cropped card image.

        Returns:
            Extracted rank region.
        """
        h, w = card_image.shape[:2]
        rx, ry, rw, rh = self.rank_region

        x1 = int(w * rx)
        y1 = int(h * ry)
        x2 = int(w * (rx + rw))
        y2 = int(h * (ry + rh))

        return card_image[y1:y2, x1:x2]

    def _extract_suit_region(self, card_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract the suit region from a card image.

        Args:
            card_image: Cropped card image.

        Returns:
            Extracted suit region.
        """
        h, w = card_image.shape[:2]
        sx, sy, sw, sh = self.suit_region

        x1 = int(w * sx)
        y1 = int(h * sy)
        x2 = int(w * (sx + sw))
        y2 = int(h * (sy + sh))

        return card_image[y1:y2, x1:x2]

    def _identify_rank(
        self, card_image: NDArray[np.uint8]
    ) -> tuple[Rank, float] | None:
        """Identify the rank of a card.

        Args:
            card_image: Cropped card image.

        Returns:
            Tuple of (Rank, confidence) or None if no match.
        """
        rank_region = self._extract_rank_region(card_image)

        if rank_region.size == 0:
            return None

        best_rank: Rank | None = None
        best_confidence = 0.0

        for rank, template_name in RANK_TEMPLATES.items():
            matches = self.template_matcher.match_template(
                rank_region, template_name, threshold=0.5
            )
            if matches and matches[0].confidence > best_confidence:
                best_rank = rank
                best_confidence = matches[0].confidence

        if best_rank is None:
            return None

        return (best_rank, best_confidence)

    def _identify_suit(
        self, card_image: NDArray[np.uint8]
    ) -> tuple[Suit, float] | None:
        """Identify the suit of a card.

        Args:
            card_image: Cropped card image.

        Returns:
            Tuple of (Suit, confidence) or None if no match.
        """
        suit_region = self._extract_suit_region(card_image)

        if suit_region.size == 0:
            return None

        best_suit: Suit | None = None
        best_confidence = 0.0

        for suit, template_name in SUIT_TEMPLATES.items():
            matches = self.template_matcher.match_template(
                suit_region, template_name, threshold=0.5
            )
            if matches and matches[0].confidence > best_confidence:
                best_suit = suit
                best_confidence = matches[0].confidence

        if best_suit is None:
            return None

        return (best_suit, best_confidence)

    def _is_card_selected(self, card_y: int, avg_y: float) -> bool:
        """Check if a card is selected based on Y position.

        Selected cards in Balatro appear raised (lower Y value).

        Args:
            card_y: Y position of the card.
            avg_y: Average Y position of all cards.

        Returns:
            True if card appears selected.
        """
        return (avg_y - card_y) > self.selection_threshold

    def detect_hand(
        self,
        image: NDArray[np.uint8],
        hand_region: tuple[int, int, int, int] | None = None,
    ) -> list[DetectedCard]:
        """Detect cards in the hand region.

        Args:
            image: Full game screenshot.
            hand_region: Optional (x, y, w, h) tuple defining hand area.
                        If None, uses bottom third of screen.

        Returns:
            List of DetectedCard objects in the hand, sorted left-to-right.
        """
        h, w = image.shape[:2]

        if hand_region is None:
            # Default to bottom third of screen
            hand_region = (0, int(h * 0.66), w, int(h * 0.34))

        x, y, region_w, region_h = hand_region
        hand_image = image[y : y + region_h, x : x + region_w]

        cards = self.detect_cards(hand_image)

        # Adjust coordinates to full image
        for card in cards:
            card.x += x
            card.y += y

        return cards

    def get_selected_cards(self, cards: list[DetectedCard]) -> list[DetectedCard]:
        """Filter to only selected cards.

        Args:
            cards: List of detected cards.

        Returns:
            List of cards that are selected.
        """
        return [c for c in cards if c.selected]

    def count_by_rank(self, cards: list[DetectedCard]) -> dict[Rank, int]:
        """Count cards by rank.

        Args:
            cards: List of detected cards.

        Returns:
            Dictionary mapping Rank to count.
        """
        counts: dict[Rank, int] = {}
        for card in cards:
            counts[card.rank] = counts.get(card.rank, 0) + 1
        return counts

    def count_by_suit(self, cards: list[DetectedCard]) -> dict[Suit, int]:
        """Count cards by suit.

        Args:
            cards: List of detected cards.

        Returns:
            Dictionary mapping Suit to count.
        """
        counts: dict[Suit, int] = {}
        for card in cards:
            counts[card.suit] = counts.get(card.suit, 0) + 1
        return counts
