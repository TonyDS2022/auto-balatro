"""Input backend abstraction for mouse/keyboard control.

Provides a swappable backend interface so we can:
- Use pyautogui for real game interaction
- Use a mock backend for testing
- Support other backends (e.g., direct window messages)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class InputEvent:
    """Record of an input event for testing/replay."""

    event_type: str  # "move", "click", "drag", "scroll"
    x: int = 0
    y: int = 0
    button: str = "left"
    clicks: int = 1
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class InputBackend(ABC):
    """Abstract base class for input backends."""

    @abstractmethod
    def get_position(self) -> tuple[int, int]:
        """Get current cursor position."""
        pass

    @abstractmethod
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        """Move cursor to position."""
        pass

    @abstractmethod
    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Click at position (or current position if None)."""
        pass

    @abstractmethod
    def drag_to(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
        button: str = "left",
    ) -> None:
        """Drag from current position to target."""
        pass

    @abstractmethod
    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        """Scroll at position."""
        pass


class PyAutoGUIBackend(InputBackend):
    """Real input backend using pyautogui."""

    def __init__(self) -> None:
        """Initialize pyautogui backend."""
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        self._pyautogui = pyautogui
        logger.info("PyAutoGUIBackend initialized")

    def get_position(self) -> tuple[int, int]:
        """Get current cursor position."""
        return self._pyautogui.position()

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        """Move cursor to position."""
        self._pyautogui.moveTo(x, y, duration=duration, tween=self._pyautogui.easeOutQuad)

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Click at position."""
        if x is not None and y is not None:
            self._pyautogui.click(x, y, button=button, clicks=clicks)
        else:
            self._pyautogui.click(button=button, clicks=clicks)

    def drag_to(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
        button: str = "left",
    ) -> None:
        """Drag to position."""
        curr_x, curr_y = self.get_position()
        self._pyautogui.drag(x - curr_x, y - curr_y, duration=duration, button=button)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        """Scroll at position."""
        if x is not None and y is not None:
            self._pyautogui.scroll(clicks, x, y)
        else:
            self._pyautogui.scroll(clicks)


class MockInputBackend(InputBackend):
    """Mock input backend for testing.

    Records all input events and allows inspection.
    """

    def __init__(self, initial_position: tuple[int, int] = (0, 0)) -> None:
        """Initialize mock backend.

        Args:
            initial_position: Starting cursor position.
        """
        self._position = initial_position
        self._events: list[InputEvent] = []
        logger.info("MockInputBackend initialized")

    @property
    def events(self) -> list[InputEvent]:
        """Get recorded events."""
        return self._events

    @property
    def position(self) -> tuple[int, int]:
        """Get current position."""
        return self._position

    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events.clear()

    def get_position(self) -> tuple[int, int]:
        """Get current cursor position."""
        return self._position

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        """Record move and update position."""
        self._events.append(InputEvent(
            event_type="move",
            x=x,
            y=y,
            duration=duration,
        ))
        self._position = (x, y)

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Record click event."""
        if x is not None and y is not None:
            self._position = (x, y)

        self._events.append(InputEvent(
            event_type="click",
            x=self._position[0],
            y=self._position[1],
            button=button,
            clicks=clicks,
        ))

    def drag_to(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
        button: str = "left",
    ) -> None:
        """Record drag event."""
        self._events.append(InputEvent(
            event_type="drag",
            x=x,
            y=y,
            button=button,
            duration=duration,
        ))
        self._position = (x, y)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        """Record scroll event."""
        if x is not None and y is not None:
            self._position = (x, y)

        self._events.append(InputEvent(
            event_type="scroll",
            x=self._position[0],
            y=self._position[1],
            clicks=clicks,
        ))


# Factory function for creating backends
def create_backend(backend_type: str = "auto") -> InputBackend:
    """Create an input backend.

    Args:
        backend_type: "pyautogui", "mock", or "auto" (tries pyautogui, falls back to mock).

    Returns:
        InputBackend instance.
    """
    if backend_type == "mock":
        return MockInputBackend()

    if backend_type == "pyautogui":
        return PyAutoGUIBackend()

    # Auto: try pyautogui, fall back to mock
    try:
        return PyAutoGUIBackend()
    except Exception as e:
        logger.warning(f"PyAutoGUI unavailable ({e}), using mock backend")
        return MockInputBackend()
