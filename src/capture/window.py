"""Window management utilities for finding and tracking game windows.

This module provides cross-platform window detection and management for
the Auto-Balatro project.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a detected window.

    Attributes:
        handle: Platform-specific window handle.
        title: Window title string.
        left: Left edge X coordinate.
        top: Top edge Y coordinate.
        width: Window width in pixels.
        height: Window height in pixels.
    """

    handle: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Get right edge X coordinate."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Get bottom edge Y coordinate."""
        return self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of window."""
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (left, top, width, height)."""
        return (self.left, self.top, self.width, self.height)


class WindowManager:
    """Cross-platform window management for finding game windows.

    Automatically selects the appropriate backend based on the current
    platform (Windows, Linux, or macOS).
    """

    def __init__(self) -> None:
        """Initialize the window manager with platform-specific backend."""
        self._platform = platform.system()
        self._find_func: Callable[[str], WindowInfo | None] = self._get_find_function()

    def _get_find_function(self) -> Callable[[str], WindowInfo | None]:
        """Get the platform-appropriate window finding function.

        Returns:
            Function that takes a window title and returns WindowInfo or None.
        """
        if self._platform == "Windows":
            return self._find_window_windows
        elif self._platform == "Linux":
            return self._find_window_linux
        elif self._platform == "Darwin":
            return self._find_window_macos
        else:
            logger.warning(f"Unknown platform: {self._platform}, using fallback")
            return self._find_window_fallback

    def find_window(self, title: str) -> WindowInfo | None:
        """Find a window by its title.

        Args:
            title: Window title to search for (partial match supported).

        Returns:
            WindowInfo if found, None otherwise.
        """
        try:
            return self._find_func(title)
        except Exception as e:
            logger.error(f"Error finding window '{title}': {e}")
            return None

    def find_all_windows(self, title: str) -> list[WindowInfo]:
        """Find all windows matching the title.

        Args:
            title: Window title to search for (partial match supported).

        Returns:
            List of WindowInfo for all matching windows.
        """
        if self._platform == "Windows":
            return self._find_all_windows_windows(title)
        elif self._platform == "Linux":
            return self._find_all_windows_linux(title)
        else:
            # Fallback: return single window if found
            window = self.find_window(title)
            return [window] if window else []

    def _find_window_windows(self, title: str) -> WindowInfo | None:
        """Find window on Windows using win32gui.

        Args:
            title: Window title to search for.

        Returns:
            WindowInfo if found, None otherwise.
        """
        try:
            import win32gui

            def callback(hwnd: int, results: list) -> bool:
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title.lower() in window_title.lower():
                        results.append((hwnd, window_title))
                return True

            results: list[tuple[int, str]] = []
            win32gui.EnumWindows(callback, results)

            if not results:
                return None

            # Return first match
            hwnd, window_title = results[0]
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect

            return WindowInfo(
                handle=hwnd,
                title=window_title,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )

        except ImportError:
            logger.warning("win32gui not available, trying pyautogui fallback")
            return self._find_window_fallback(title)
        except Exception as e:
            logger.error(f"Windows window search error: {e}")
            return None

    def _find_all_windows_windows(self, title: str) -> list[WindowInfo]:
        """Find all windows on Windows matching the title.

        Args:
            title: Window title to search for.

        Returns:
            List of WindowInfo for all matching windows.
        """
        try:
            import win32gui

            def callback(hwnd: int, results: list) -> bool:
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title.lower() in window_title.lower():
                        results.append((hwnd, window_title))
                return True

            results: list[tuple[int, str]] = []
            win32gui.EnumWindows(callback, results)

            windows = []
            for hwnd, window_title in results:
                rect = win32gui.GetWindowRect(hwnd)
                left, top, right, bottom = rect
                windows.append(
                    WindowInfo(
                        handle=hwnd,
                        title=window_title,
                        left=left,
                        top=top,
                        width=right - left,
                        height=bottom - top,
                    )
                )
            return windows

        except ImportError:
            logger.warning("win32gui not available")
            return []
        except Exception as e:
            logger.error(f"Windows window enumeration error: {e}")
            return []

    def _find_window_linux(self, title: str) -> WindowInfo | None:
        """Find window on Linux using wmctrl or xdotool.

        Args:
            title: Window title to search for.

        Returns:
            WindowInfo if found, None otherwise.
        """
        # Try xdotool first (more reliable for getting geometry)
        try:
            # Find window ID
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0 or not result.stdout.strip():
                return self._find_window_linux_wmctrl(title)

            window_ids = result.stdout.strip().split("\n")
            if not window_ids:
                return None

            window_id = window_ids[0]

            # Get window geometry
            geo_result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if geo_result.returncode != 0:
                return None

            # Parse geometry output
            geo = {}
            for line in geo_result.stdout.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    geo[key] = int(value)

            # Get window name
            name_result = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            window_title = name_result.stdout.strip() if name_result.returncode == 0 else title

            return WindowInfo(
                handle=int(window_id),
                title=window_title,
                left=geo.get("X", 0),
                top=geo.get("Y", 0),
                width=geo.get("WIDTH", 0),
                height=geo.get("HEIGHT", 0),
            )

        except FileNotFoundError:
            return self._find_window_linux_wmctrl(title)
        except subprocess.TimeoutExpired:
            logger.warning("xdotool timed out")
            return None
        except Exception as e:
            logger.error(f"Linux xdotool error: {e}")
            return self._find_window_linux_wmctrl(title)

    def _find_window_linux_wmctrl(self, title: str) -> WindowInfo | None:
        """Find window on Linux using wmctrl.

        Args:
            title: Window title to search for.

        Returns:
            WindowInfo if found, None otherwise.
        """
        try:
            result = subprocess.run(
                ["wmctrl", "-l", "-G"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            for line in result.stdout.strip().split("\n"):
                if title.lower() in line.lower():
                    parts = line.split()
                    if len(parts) >= 8:
                        # Format: window_id desktop x y width height hostname title...
                        window_id = int(parts[0], 16)
                        x = int(parts[2])
                        y = int(parts[3])
                        width = int(parts[4])
                        height = int(parts[5])
                        window_title = " ".join(parts[7:])

                        return WindowInfo(
                            handle=window_id,
                            title=window_title,
                            left=x,
                            top=y,
                            width=width,
                            height=height,
                        )

            return None

        except FileNotFoundError:
            logger.warning("wmctrl not found, install with: sudo apt install wmctrl")
            return None
        except Exception as e:
            logger.error(f"Linux wmctrl error: {e}")
            return None

    def _find_all_windows_linux(self, title: str) -> list[WindowInfo]:
        """Find all windows on Linux matching the title.

        Args:
            title: Window title to search for.

        Returns:
            List of WindowInfo for all matching windows.
        """
        windows = []
        try:
            result = subprocess.run(
                ["wmctrl", "-l", "-G"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return windows

            for line in result.stdout.strip().split("\n"):
                if title.lower() in line.lower():
                    parts = line.split()
                    if len(parts) >= 8:
                        window_id = int(parts[0], 16)
                        x = int(parts[2])
                        y = int(parts[3])
                        width = int(parts[4])
                        height = int(parts[5])
                        window_title = " ".join(parts[7:])

                        windows.append(
                            WindowInfo(
                                handle=window_id,
                                title=window_title,
                                left=x,
                                top=y,
                                width=width,
                                height=height,
                            )
                        )

        except Exception as e:
            logger.error(f"Linux window enumeration error: {e}")

        return windows

    def _find_window_macos(self, title: str) -> WindowInfo | None:
        """Find window on macOS using Quartz.

        Args:
            title: Window title to search for.

        Returns:
            WindowInfo if found, None otherwise.
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )

            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )

            for window in window_list:
                window_name = window.get("kCGWindowName", "")
                owner_name = window.get("kCGWindowOwnerName", "")

                if title.lower() in str(window_name).lower() or title.lower() in str(
                    owner_name
                ).lower():
                    bounds = window.get("kCGWindowBounds", {})

                    return WindowInfo(
                        handle=window.get("kCGWindowNumber", 0),
                        title=window_name or owner_name,
                        left=int(bounds.get("X", 0)),
                        top=int(bounds.get("Y", 0)),
                        width=int(bounds.get("Width", 0)),
                        height=int(bounds.get("Height", 0)),
                    )

            return None

        except ImportError:
            logger.warning("Quartz not available, using pyautogui fallback")
            return self._find_window_fallback(title)
        except Exception as e:
            logger.error(f"macOS window search error: {e}")
            return None

    def _find_window_fallback(self, title: str) -> WindowInfo | None:
        """Fallback window finding using pyautogui.

        Args:
            title: Window title to search for.

        Returns:
            WindowInfo if found, None otherwise.
        """
        try:
            import pyautogui

            windows = pyautogui.getWindowsWithTitle(title)
            if not windows:
                return None

            window = windows[0]
            return WindowInfo(
                handle=0,  # pyautogui doesn't provide handle
                title=window.title,
                left=window.left,
                top=window.top,
                width=window.width,
                height=window.height,
            )

        except Exception as e:
            logger.error(f"Fallback window search error: {e}")
            return None

    def bring_to_front(self, window: WindowInfo) -> bool:
        """Bring a window to the foreground.

        Args:
            window: WindowInfo of window to bring to front.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self._platform == "Windows":
                import win32gui

                win32gui.SetForegroundWindow(window.handle)
                return True
            elif self._platform == "Linux":
                subprocess.run(
                    ["wmctrl", "-i", "-a", hex(window.handle)],
                    timeout=5,
                )
                return True
            elif self._platform == "Darwin":
                # macOS requires AppleScript for window focus
                import pyautogui

                windows = pyautogui.getWindowsWithTitle(window.title)
                if windows:
                    windows[0].activate()
                    return True
                return False
            else:
                return False

        except Exception as e:
            logger.error(f"Error bringing window to front: {e}")
            return False

    def is_window_visible(self, window: WindowInfo) -> bool:
        """Check if a window is currently visible.

        Args:
            window: WindowInfo to check.

        Returns:
            True if window is visible, False otherwise.
        """
        try:
            if self._platform == "Windows":
                import win32gui

                return bool(win32gui.IsWindowVisible(window.handle))
            else:
                # On other platforms, re-find the window
                found = self.find_window(window.title)
                return found is not None

        except Exception as e:
            logger.error(f"Error checking window visibility: {e}")
            return False
