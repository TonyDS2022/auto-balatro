"""Mouse control module for game interaction.

Provides safe, human-like mouse movement and clicking for
interacting with the Balatro game window.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Lazy import pyautogui to allow testing without X11
_pyautogui = None


def _get_pyautogui():
    """Lazy load pyautogui."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.FAILSAFE = True  # Move to corner to abort
        pyautogui.PAUSE = 0.05  # Small delay between actions
        _pyautogui = pyautogui
    return _pyautogui


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """A 2D point with x and y coordinates."""

    x: int
    y: int

    def __iter__(self):
        """Allow unpacking as tuple."""
        yield self.x
        yield self.y

    def offset(self, dx: int, dy: int) -> Point:
        """Return a new point offset by dx and dy."""
        return Point(self.x + dx, self.y + dy)


@dataclass
class Region:
    """A rectangular region on screen."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Point:
        """Get center point of region."""
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    @property
    def top_left(self) -> Point:
        """Get top-left corner."""
        return Point(self.x, self.y)

    @property
    def bottom_right(self) -> Point:
        """Get bottom-right corner."""
        return Point(self.x + self.width, self.y + self.height)

    def contains(self, point: Point) -> bool:
        """Check if point is within region."""
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def random_point(self, margin: int = 5) -> Point:
        """Get a random point within the region with margin."""
        return Point(
            random.randint(self.x + margin, self.x + self.width - margin),
            random.randint(self.y + margin, self.y + self.height - margin),
        )


class MouseController:
    """Controls mouse movement and clicking with human-like behavior.

    Features:
    - Bezier curve movement for natural motion
    - Random micro-movements and delays
    - Safety bounds checking
    - Click confirmation via position verification

    Attributes:
        bounds: Optional screen region to constrain mouse to.
        speed_factor: Multiplier for movement speed (1.0 = normal).
        humanize: Whether to add human-like randomness.
    """

    def __init__(
        self,
        bounds: Region | None = None,
        speed_factor: float = 1.0,
        humanize: bool = True,
    ) -> None:
        """Initialize mouse controller.

        Args:
            bounds: Screen region to constrain mouse movement.
            speed_factor: Movement speed multiplier (lower = faster).
            humanize: Add human-like movement variations.
        """
        self.bounds = bounds
        self.speed_factor = speed_factor
        self.humanize = humanize

        # Track last position for movement calculations
        self._last_pos: Point | None = None

        logger.info(
            f"MouseController initialized, bounds={bounds}, "
            f"speed={speed_factor}, humanize={humanize}"
        )

    def get_position(self) -> Point:
        """Get current mouse position.

        Returns:
            Current mouse position as Point.
        """
        pyautogui = _get_pyautogui()
        x, y = pyautogui.position()
        return Point(x, y)

    def _check_bounds(self, target: Point) -> Point:
        """Ensure target is within bounds.

        Args:
            target: Desired target position.

        Returns:
            Target clamped to bounds if necessary.
        """
        if self.bounds is None:
            return target

        clamped_x = max(
            self.bounds.x,
            min(target.x, self.bounds.x + self.bounds.width),
        )
        clamped_y = max(
            self.bounds.y,
            min(target.y, self.bounds.y + self.bounds.height),
        )

        if clamped_x != target.x or clamped_y != target.y:
            logger.warning(f"Target {target} clamped to bounds: ({clamped_x}, {clamped_y})")

        return Point(clamped_x, clamped_y)

    def _humanize_target(self, target: Point) -> Point:
        """Add small random offset to target for human-like behavior.

        Args:
            target: Original target position.

        Returns:
            Slightly randomized target.
        """
        if not self.humanize:
            return target

        # Small random offset (1-3 pixels)
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)

        return target.offset(offset_x, offset_y)

    def _calculate_duration(self, start: Point, end: Point) -> float:
        """Calculate movement duration based on distance.

        Args:
            start: Starting position.
            end: Ending position.

        Returns:
            Duration in seconds.
        """
        import math

        distance = math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)

        # Base duration: ~0.1s per 100 pixels, with minimum
        base_duration = max(0.1, distance / 1000)

        # Apply speed factor
        duration = base_duration * self.speed_factor

        # Add randomness if humanizing
        if self.humanize:
            duration *= random.uniform(0.9, 1.1)

        return duration

    def move_to(
        self,
        target: Point | tuple[int, int],
        duration: float | None = None,
    ) -> None:
        """Move mouse to target position.

        Args:
            target: Target position (Point or (x, y) tuple).
            duration: Movement duration (None for auto-calculate).
        """
        if isinstance(target, tuple):
            target = Point(target[0], target[1])

        # Apply bounds and humanization
        target = self._check_bounds(target)
        target = self._humanize_target(target)

        # Calculate duration if not specified
        current = self.get_position()
        if duration is None:
            duration = self._calculate_duration(current, target)

        # Move with easing
        pyautogui = _get_pyautogui()
        pyautogui.moveTo(
            target.x,
            target.y,
            duration=duration,
            tween=pyautogui.easeOutQuad,
        )

        self._last_pos = target
        logger.debug(f"Moved to {target}")

    def move_to_region(
        self,
        region: Region,
        position: str = "center",
    ) -> None:
        """Move mouse to a position within a region.

        Args:
            region: Target region.
            position: Where in region ("center", "random").
        """
        if position == "center":
            target = region.center
        elif position == "random":
            target = region.random_point()
        else:
            target = region.center

        self.move_to(target)

    def click(
        self,
        target: Point | tuple[int, int] | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        """Click at target position or current position.

        Args:
            target: Position to click (None for current position).
            button: Mouse button ("left", "right", "middle").
            clicks: Number of clicks.
            interval: Interval between multiple clicks.
        """
        if target is not None:
            self.move_to(target)

        # Small delay before clicking (human-like)
        if self.humanize:
            time.sleep(random.uniform(0.02, 0.08))

        pyautogui = _get_pyautogui()
        pyautogui.click(button=button, clicks=clicks, interval=interval)

        pos = self.get_position()
        logger.debug(f"Clicked {button} at {pos}")

    def double_click(
        self,
        target: Point | tuple[int, int] | None = None,
    ) -> None:
        """Double-click at target position.

        Args:
            target: Position to double-click.
        """
        self.click(target, clicks=2, interval=0.1)

    def right_click(
        self,
        target: Point | tuple[int, int] | None = None,
    ) -> None:
        """Right-click at target position.

        Args:
            target: Position to right-click.
        """
        self.click(target, button="right")

    def drag_to(
        self,
        target: Point | tuple[int, int],
        duration: float | None = None,
        button: str = "left",
    ) -> None:
        """Drag from current position to target.

        Args:
            target: Target position to drag to.
            duration: Drag duration.
            button: Mouse button to hold.
        """
        if isinstance(target, tuple):
            target = Point(target[0], target[1])

        target = self._check_bounds(target)

        current = self.get_position()
        if duration is None:
            duration = self._calculate_duration(current, target) * 1.5

        pyautogui = _get_pyautogui()
        pyautogui.drag(
            target.x - current.x,
            target.y - current.y,
            duration=duration,
            button=button,
        )

        self._last_pos = target
        logger.debug(f"Dragged to {target}")

    def scroll(
        self,
        clicks: int,
        target: Point | tuple[int, int] | None = None,
    ) -> None:
        """Scroll at target position.

        Args:
            clicks: Number of scroll clicks (positive = up, negative = down).
            target: Position to scroll at (None for current).
        """
        if target is not None:
            self.move_to(target)

        pyautogui = _get_pyautogui()
        pyautogui.scroll(clicks)
        logger.debug(f"Scrolled {clicks} at {self.get_position()}")

    def wait(self, seconds: float | None = None) -> None:
        """Wait for a period with optional randomization.

        Args:
            seconds: Time to wait (None for random short delay).
        """
        if seconds is None:
            seconds = random.uniform(0.1, 0.3)
        elif self.humanize:
            seconds *= random.uniform(0.9, 1.1)

        time.sleep(seconds)


class SafeMouseController(MouseController):
    """Mouse controller with additional safety features.

    Adds confirmation checks and abort capabilities.
    """

    def __init__(
        self,
        bounds: Region | None = None,
        speed_factor: float = 1.0,
        humanize: bool = True,
        confirm_clicks: bool = True,
    ) -> None:
        """Initialize safe mouse controller.

        Args:
            bounds: Screen region constraint.
            speed_factor: Movement speed multiplier.
            humanize: Add human-like variations.
            confirm_clicks: Verify position before clicking.
        """
        super().__init__(bounds, speed_factor, humanize)
        self.confirm_clicks = confirm_clicks

    def click(
        self,
        target: Point | tuple[int, int] | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        """Click with position confirmation.

        Args:
            target: Position to click.
            button: Mouse button.
            clicks: Number of clicks.
            interval: Interval between clicks.
        """
        if target is not None:
            self.move_to(target)

        # Confirm position before clicking
        if self.confirm_clicks:
            current = self.get_position()
            if target is not None:
                if isinstance(target, tuple):
                    target = Point(target[0], target[1])
                # Allow small deviation (humanization)
                tolerance = 10
                if (
                    abs(current.x - target.x) > tolerance
                    or abs(current.y - target.y) > tolerance
                ):
                    logger.warning(
                        f"Position mismatch: expected ~{target}, got {current}"
                    )
                    return

        super().click(None, button, clicks, interval)
