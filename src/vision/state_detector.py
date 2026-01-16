"""Game state detection module for Auto-Balatro.

This module provides functionality to detect the current game phase by analyzing
screen captures for UI indicators, buttons, and other visual elements specific
to each game state.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.game.constants import GamePhase

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Default template directory path
DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "data" / "templates" / "ui"


@runtime_checkable
class TemplateMatcher(Protocol):
    """Protocol for template matching implementations.

    This protocol defines the interface that any template matcher must implement
    to be used with the StateDetector.
    """

    def match(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        threshold: float = 0.8,
    ) -> tuple[bool, tuple[int, int] | None, float]:
        """Match a template against an image.

        Args:
            image: BGR image to search in.
            template_name: Name of the template file (without path).
            threshold: Minimum confidence threshold for a match.

        Returns:
            Tuple of (matched, center_coords, confidence).
            - matched: True if template was found above threshold.
            - center_coords: (x, y) center of match if found, None otherwise.
            - confidence: Confidence score of the best match.
        """
        ...

    def match_multiple(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        threshold: float = 0.8,
        max_matches: int = 10,
    ) -> list[tuple[tuple[int, int], float]]:
        """Find multiple occurrences of a template in an image.

        Args:
            image: BGR image to search in.
            template_name: Name of the template file.
            threshold: Minimum confidence threshold.
            max_matches: Maximum number of matches to return.

        Returns:
            List of ((x, y), confidence) tuples for each match found.
        """
        ...


class StateDetector:
    """Detects the current game phase by analyzing UI indicators.

    This class analyzes screen captures to determine which phase of the game
    is currently active. It uses template matching for UI element detection
    and maintains a history buffer to provide stable state detection.

    Attributes:
        template_matcher: Optional template matcher for UI detection.
        stability_frames: Number of consistent frames required to confirm state change.
    """

    # Template file names for each detectable UI element
    TEMPLATES = {
        # Main menu templates
        "button_play": "button_play.png",
        "title_screen": "title_screen.png",
        # Blind selection templates
        "ui_blind_select": "ui_blind_select.png",
        "button_skip": "button_skip.png",
        "blind_small": "blind_small.png",
        "blind_big": "blind_big.png",
        "blind_boss": "blind_boss.png",
        # Playing phase templates
        "button_play_hand": "button_play_hand.png",
        "button_discard": "button_discard.png",
        "hand_area": "hand_area.png",
        # Scoring templates
        "scoring_animation": "scoring_animation.png",
        "chips_display": "chips_display.png",
        # Shop templates
        "ui_shop": "ui_shop.png",
        "button_next_round": "button_next_round.png",
        "shop_reroll": "shop_reroll.png",
        # Game over templates
        "ui_game_over": "ui_game_over.png",
        "button_main_menu": "button_main_menu.png",
        "final_score": "final_score.png",
        # Pause templates
        "ui_pause_menu": "ui_pause_menu.png",
        "button_resume": "button_resume.png",
        # Round info templates
        "round_complete": "round_complete.png",
    }

    # Button names that can be detected
    BUTTON_TEMPLATES = [
        "button_play",
        "button_play_hand",
        "button_discard",
        "button_skip",
        "button_next_round",
        "button_main_menu",
        "button_resume",
        "shop_reroll",
    ]

    def __init__(
        self,
        template_matcher: TemplateMatcher | None = None,
        stability_frames: int = 3,
    ) -> None:
        """Initialize the state detector.

        Args:
            template_matcher: Optional template matcher for UI element detection.
                If None, detection methods will use fallback color/region analysis.
            stability_frames: Number of consistent detections required before
                confirming a state change. Higher values reduce flickering but
                increase detection latency.
        """
        self.template_matcher = template_matcher
        self.stability_frames = max(1, stability_frames)

        # State history buffer for stability checking
        self._state_history: deque[GamePhase] = deque(maxlen=stability_frames)
        self._last_stable_phase: GamePhase = GamePhase.MAIN_MENU

        # Cache for detected button locations
        self._button_cache: dict[str, tuple[int, int]] = {}
        self._button_cache_frame_id: int = -1
        self._current_frame_id: int = 0

        logger.debug(
            f"StateDetector initialized with stability_frames={stability_frames}, "
            f"template_matcher={'provided' if template_matcher else 'None'}"
        )

    def detect_phase(self, image: NDArray[np.uint8]) -> GamePhase:
        """Analyze image to determine the current game phase.

        Checks for phase-specific UI elements in priority order to determine
        which game phase is currently active.

        Args:
            image: BGR numpy array of the current game screen.

        Returns:
            The detected GamePhase enum value.

        Note:
            If no phase can be confidently detected, returns the last known
            stable phase to prevent erratic behavior.
        """
        if image is None or image.size == 0:
            logger.warning("detect_phase received empty image")
            return self._last_stable_phase

        self._current_frame_id += 1

        # Check phases in order of priority (most distinctive first)
        # Pause menu overlays other screens, check first
        if self._detect_pause(image):
            return GamePhase.PAUSE

        # Game over is distinctive
        if self._detect_game_over(image):
            return GamePhase.GAME_OVER

        # Scoring animation is transient
        if self._detect_scoring(image):
            return GamePhase.SCORING

        # Shop has unique elements
        if self._detect_shop(image):
            return GamePhase.SHOP

        # Blind selection screen
        if self._detect_blind_select(image):
            return GamePhase.BLIND_SELECT

        # Active gameplay
        if self._detect_playing(image):
            return GamePhase.PLAYING

        # Main menu (fallback)
        if self._detect_main_menu(image):
            return GamePhase.MAIN_MENU

        # If nothing detected, return last known phase
        logger.debug("No phase detected, returning last stable phase")
        return self._last_stable_phase

    def detect_phase_stable(self, image: NDArray[np.uint8]) -> GamePhase:
        """Detect game phase with stability checking.

        Only returns a new phase after it has been consistently detected
        for stability_frames consecutive frames. This prevents flickering
        between states due to transient visual artifacts.

        Args:
            image: BGR numpy array of the current game screen.

        Returns:
            The stable GamePhase enum value.
        """
        current_phase = self.detect_phase(image)
        self._state_history.append(current_phase)

        # Check if all recent detections are consistent
        if len(self._state_history) >= self.stability_frames:
            if all(phase == current_phase for phase in self._state_history):
                if current_phase != self._last_stable_phase:
                    logger.info(
                        f"Phase transition: {self._last_stable_phase.name} -> {current_phase.name}"
                    )
                self._last_stable_phase = current_phase

        return self._last_stable_phase

    def _detect_main_menu(self, image: NDArray[np.uint8]) -> bool:
        """Detect if the main menu screen is displayed.

        Looks for the "Play" button and title screen elements.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if main menu indicators are found.
        """
        if self.template_matcher is not None:
            # Try to find Play button
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["button_play"], threshold=0.7
            )
            if matched:
                logger.debug(f"Main menu detected (Play button, conf={confidence:.2f})")
                return True

            # Try title screen
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["title_screen"], threshold=0.7
            )
            if matched:
                logger.debug(f"Main menu detected (title, conf={confidence:.2f})")
                return True

        # Fallback: analyze color distribution for menu characteristics
        return self._fallback_detect_main_menu(image)

    def _detect_blind_select(self, image: NDArray[np.uint8]) -> bool:
        """Detect if the blind selection screen is displayed.

        Looks for blind options (Small/Big/Boss) and Skip button.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if blind selection UI is found.
        """
        if self.template_matcher is not None:
            # Check for blind select UI
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["ui_blind_select"], threshold=0.7
            )
            if matched:
                logger.debug(f"Blind select detected (UI, conf={confidence:.2f})")
                return True

            # Check for Skip button
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["button_skip"], threshold=0.7
            )
            if matched:
                logger.debug(f"Blind select detected (Skip button, conf={confidence:.2f})")
                return True

        # Fallback detection
        return self._fallback_detect_blind_select(image)

    def _detect_playing(self, image: NDArray[np.uint8]) -> bool:
        """Detect if active gameplay is in progress.

        Looks for hand area with cards, Play Hand and Discard buttons.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if playing phase indicators are found.
        """
        if self.template_matcher is not None:
            # Check for Play Hand button
            matched_play, _, conf_play = self.template_matcher.match(
                image, self.TEMPLATES["button_play_hand"], threshold=0.7
            )

            # Check for Discard button
            matched_discard, _, conf_discard = self.template_matcher.match(
                image, self.TEMPLATES["button_discard"], threshold=0.7
            )

            if matched_play or matched_discard:
                logger.debug(
                    f"Playing phase detected (Play={matched_play}, Discard={matched_discard})"
                )
                return True

        # Fallback detection
        return self._fallback_detect_playing(image)

    def _detect_scoring(self, image: NDArray[np.uint8]) -> bool:
        """Detect if scoring animation is displayed.

        Looks for chips flying animation and score incrementing display.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if scoring animation is found.
        """
        if self.template_matcher is not None:
            # Check for scoring animation
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["scoring_animation"], threshold=0.6
            )
            if matched:
                logger.debug(f"Scoring detected (animation, conf={confidence:.2f})")
                return True

            # Check for chips display during scoring
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["chips_display"], threshold=0.6
            )
            if matched:
                logger.debug(f"Scoring detected (chips, conf={confidence:.2f})")
                return True

        # Fallback: look for scoring visual characteristics
        return self._fallback_detect_scoring(image)

    def _detect_shop(self, image: NDArray[np.uint8]) -> bool:
        """Detect if the shop screen is displayed.

        Looks for shop UI elements, joker cards for sale, Next Round button.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if shop UI is found.
        """
        if self.template_matcher is not None:
            # Check for shop UI
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["ui_shop"], threshold=0.7
            )
            if matched:
                logger.debug(f"Shop detected (UI, conf={confidence:.2f})")
                return True

            # Check for Next Round button
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["button_next_round"], threshold=0.7
            )
            if matched:
                logger.debug(f"Shop detected (Next Round, conf={confidence:.2f})")
                return True

        # Fallback detection
        return self._fallback_detect_shop(image)

    def _detect_game_over(self, image: NDArray[np.uint8]) -> bool:
        """Detect if the game over screen is displayed.

        Looks for game over UI, Main Menu button, final score display.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if game over screen is found.
        """
        if self.template_matcher is not None:
            # Check for game over UI
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["ui_game_over"], threshold=0.7
            )
            if matched:
                logger.debug(f"Game over detected (UI, conf={confidence:.2f})")
                return True

            # Check for Main Menu button
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["button_main_menu"], threshold=0.7
            )
            if matched:
                logger.debug(f"Game over detected (Main Menu, conf={confidence:.2f})")
                return True

        # Fallback detection
        return self._fallback_detect_game_over(image)

    def _detect_pause(self, image: NDArray[np.uint8]) -> bool:
        """Detect if the pause menu is displayed.

        Looks for pause menu overlay and Resume button.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if pause menu is found.
        """
        if self.template_matcher is not None:
            # Check for pause menu
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["ui_pause_menu"], threshold=0.7
            )
            if matched:
                logger.debug(f"Pause menu detected (UI, conf={confidence:.2f})")
                return True

            # Check for Resume button
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["button_resume"], threshold=0.7
            )
            if matched:
                logger.debug(f"Pause menu detected (Resume, conf={confidence:.2f})")
                return True

        # Fallback: check for dark overlay characteristic of pause menu
        return self._fallback_detect_pause(image)

    def get_blind_type(self, image: NDArray[np.uint8]) -> str | None:
        """Determine the type of blind being fought in PLAYING phase.

        Args:
            image: BGR numpy array of the current game screen.

        Returns:
            "small", "big", "boss", or None if not determinable.
        """
        if self.template_matcher is not None:
            # Check for each blind type indicator
            for blind_type in ["small", "big", "boss"]:
                template_key = f"blind_{blind_type}"
                matched, _, confidence = self.template_matcher.match(
                    image, self.TEMPLATES[template_key], threshold=0.7
                )
                if matched:
                    logger.debug(f"Blind type detected: {blind_type} (conf={confidence:.2f})")
                    return blind_type

        # Fallback: try to detect from visual indicators
        return self._fallback_detect_blind_type(image)

    def is_round_complete(self, image: NDArray[np.uint8]) -> bool:
        """Check if the current blind has been beaten.

        Args:
            image: BGR numpy array of the current game screen.

        Returns:
            True if round complete indicators are found.
        """
        if self.template_matcher is not None:
            matched, _, confidence = self.template_matcher.match(
                image, self.TEMPLATES["round_complete"], threshold=0.7
            )
            if matched:
                logger.debug(f"Round complete detected (conf={confidence:.2f})")
                return True

        # Could also detect by checking if we're transitioning to shop
        return False

    def get_button_locations(
        self, image: NDArray[np.uint8]
    ) -> dict[str, tuple[int, int]]:
        """Find clickable buttons and their center coordinates.

        Args:
            image: BGR numpy array of the current game screen.

        Returns:
            Dictionary mapping button names to (x, y) center coordinates.
            Button names include: "play_hand", "discard", "next_round",
            "skip", "main_menu", "resume", "reroll".
        """
        # Use cache if same frame
        if self._button_cache_frame_id == self._current_frame_id:
            return self._button_cache.copy()

        buttons: dict[str, tuple[int, int]] = {}

        if self.template_matcher is not None:
            button_name_map = {
                "button_play": "play",
                "button_play_hand": "play_hand",
                "button_discard": "discard",
                "button_skip": "skip",
                "button_next_round": "next_round",
                "button_main_menu": "main_menu",
                "button_resume": "resume",
                "shop_reroll": "reroll",
            }

            for template_key, button_name in button_name_map.items():
                matched, coords, confidence = self.template_matcher.match(
                    image, self.TEMPLATES[template_key], threshold=0.7
                )
                if matched and coords is not None:
                    buttons[button_name] = coords
                    logger.debug(
                        f"Button '{button_name}' found at {coords} (conf={confidence:.2f})"
                    )

        # Update cache
        self._button_cache = buttons.copy()
        self._button_cache_frame_id = self._current_frame_id

        return buttons

    # =========================================================================
    # Fallback detection methods (color/region analysis when no template matcher)
    # =========================================================================

    def _fallback_detect_main_menu(self, image: NDArray[np.uint8]) -> bool:
        """Fallback main menu detection using color analysis.

        Looks for characteristic colors and regions of the main menu.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if main menu characteristics are detected.
        """
        # Main menu typically has a dark background with specific accent colors
        # This is a basic heuristic and should be tuned for Balatro's specific colors
        height, width = image.shape[:2]

        # Check center region for title text colors
        center_region = image[height // 4 : height // 2, width // 4 : 3 * width // 4]

        # Convert to HSV for color analysis
        try:
            import cv2

            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            # Look for golden/yellow colors often used in game titles
            # Hue ~25-35 for gold/yellow in HSV
            gold_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
            gold_ratio = np.count_nonzero(gold_mask) / gold_mask.size

            if gold_ratio > 0.02:  # At least 2% golden pixels
                return True
        except ImportError:
            pass

        return False

    def _fallback_detect_blind_select(self, image: NDArray[np.uint8]) -> bool:
        """Fallback blind select detection.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if blind select characteristics are detected.
        """
        # Blind select typically shows three distinct card-like regions
        # This is a placeholder for more sophisticated detection
        return False

    def _fallback_detect_playing(self, image: NDArray[np.uint8]) -> bool:
        """Fallback playing phase detection.

        Looks for card-like regions in the lower portion of the screen.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if playing characteristics are detected.
        """
        height, width = image.shape[:2]

        # Cards are typically in the lower third of the screen
        hand_region = image[2 * height // 3 :, :]

        # Look for high contrast edges (card borders)
        try:
            import cv2

            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            # Cards create significant edges
            if edge_density > 0.05:
                return True
        except ImportError:
            pass

        return False

    def _fallback_detect_scoring(self, image: NDArray[np.uint8]) -> bool:
        """Fallback scoring detection.

        Looks for bright animated regions characteristic of scoring.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if scoring characteristics are detected.
        """
        # Scoring typically has bright animated elements
        # Check for high brightness in center region
        height, width = image.shape[:2]
        center = image[height // 3 : 2 * height // 3, width // 4 : 3 * width // 4]

        # Calculate mean brightness
        mean_brightness = np.mean(center)

        # Scoring animations tend to be brighter
        if mean_brightness > 180:
            return True

        return False

    def _fallback_detect_shop(self, image: NDArray[np.uint8]) -> bool:
        """Fallback shop detection.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if shop characteristics are detected.
        """
        # Shop has specific layout with items for sale
        # This is a placeholder
        return False

    def _fallback_detect_game_over(self, image: NDArray[np.uint8]) -> bool:
        """Fallback game over detection.

        Looks for darkened/desaturated screen with centered text.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if game over characteristics are detected.
        """
        # Game over typically has a dark overlay
        mean_brightness = np.mean(image)

        if mean_brightness < 50:  # Very dark
            # Check for some text-like bright regions
            bright_pixels = np.count_nonzero(image > 150)
            bright_ratio = bright_pixels / image.size

            if 0.01 < bright_ratio < 0.1:  # Some bright text on dark bg
                return True

        return False

    def _fallback_detect_pause(self, image: NDArray[np.uint8]) -> bool:
        """Fallback pause menu detection.

        Pause typically darkens/blurs the background with overlay.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            True if pause characteristics are detected.
        """
        # Pause menu typically has semi-transparent dark overlay
        # Check for reduced overall brightness but not completely dark
        mean_brightness = np.mean(image)

        if 30 < mean_brightness < 100:
            # Calculate variance - pause screens have lower variance due to overlay
            variance = np.var(image)
            if variance < 2000:  # Low variance indicates uniform overlay
                return True

        return False

    def _fallback_detect_blind_type(self, image: NDArray[np.uint8]) -> str | None:
        """Fallback blind type detection.

        Args:
            image: BGR numpy array to analyze.

        Returns:
            Blind type string or None.
        """
        # Would need to analyze specific regions for blind indicators
        # This requires knowledge of Balatro's UI layout
        return None

    def reset_state(self) -> None:
        """Reset the detector's internal state.

        Clears the state history buffer and resets to MAIN_MENU.
        Useful when starting a new game or recovering from errors.
        """
        self._state_history.clear()
        self._last_stable_phase = GamePhase.MAIN_MENU
        self._button_cache.clear()
        self._button_cache_frame_id = -1
        logger.info("StateDetector state reset")

    @property
    def last_stable_phase(self) -> GamePhase:
        """Get the last confirmed stable game phase.

        Returns:
            The most recently confirmed stable GamePhase.
        """
        return self._last_stable_phase

    @property
    def state_history(self) -> list[GamePhase]:
        """Get the recent state detection history.

        Returns:
            List of recently detected phases (oldest to newest).
        """
        return list(self._state_history)
