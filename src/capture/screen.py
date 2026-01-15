"""Screen capture module for capturing game frames.

This module provides high-performance screen capture functionality for the
Auto-Balatro project, supporting both cross-platform (mss) and Windows-optimized
(dxcam) backends.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import mss
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ScreenCapture:
    """High-performance screen capture for game automation.

    Supports continuous background capture with frame buffering for consistent
    frame delivery regardless of processing speed.

    Attributes:
        window_title: Title of the window to capture.
        target_fps: Target frames per second for capture.
        use_dxcam: Whether to use dxcam on Windows (faster but Windows-only).
    """

    def __init__(self, config: dict) -> None:
        """Initialize the screen capture with configuration.

        Args:
            config: Configuration dictionary with keys:
                - window_title (str): Title of window to find and capture
                - target_fps (int): Target capture framerate
                - use_dxcam (bool): Use dxcam on Windows for better performance
        """
        self.window_title: str = config.get("window_title", "Balatro")
        self.target_fps: int = config.get("target_fps", 10)
        self.use_dxcam: bool = config.get("use_dxcam", False)

        # Frame timing
        self._frame_interval: float = 1.0 / self.target_fps if self.target_fps > 0 else 0

        # Capture state
        self._running: bool = False
        self._capture_thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()

        # Frame buffer (stores last N frames for smoothing)
        self._frame_buffer: deque[NDArray[np.uint8]] = deque(maxlen=3)
        self._latest_frame: NDArray[np.uint8] | None = None

        # Window bounding box cache
        self._window_bbox: tuple[int, int, int, int] | None = None
        self._bbox_last_check: float = 0.0
        self._bbox_check_interval: float = 2.0  # Re-check window position every 2 seconds

        # Backend selection
        self._dxcam_camera = None
        self._mss_instance: mss.mss | None = None
        self._use_dxcam_backend: bool = False

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the appropriate capture backend based on platform and config."""
        is_windows = platform.system() == "Windows"

        if self.use_dxcam and is_windows:
            try:
                import dxcam

                self._dxcam_camera = dxcam.create(output_color="BGR")
                self._use_dxcam_backend = True
                logger.info("Using dxcam backend for screen capture (Windows optimized)")
            except ImportError:
                logger.warning(
                    "dxcam requested but not installed. "
                    "Install with: pip install dxcam. Falling back to mss."
                )
                self._use_dxcam_backend = False
            except Exception as e:
                logger.warning(f"Failed to initialize dxcam: {e}. Falling back to mss.")
                self._use_dxcam_backend = False

        if not self._use_dxcam_backend:
            self._mss_instance = mss.mss()
            logger.info("Using mss backend for screen capture (cross-platform)")

    def find_window(self) -> tuple[int, int, int, int] | None:
        """Find the game window and return its bounding box.

        Returns:
            Tuple of (left, top, width, height) if found, None otherwise.
        """
        try:
            # Import window utilities lazily to avoid circular imports
            from .window import WindowManager

            manager = WindowManager()
            window_info = manager.find_window(self.window_title)

            if window_info is None:
                logger.debug(f"Window '{self.window_title}' not found")
                return None

            bbox = (
                window_info.left,
                window_info.top,
                window_info.width,
                window_info.height,
            )
            logger.debug(f"Found window at {bbox}")
            return bbox

        except Exception as e:
            logger.error(f"Error finding window: {e}")
            return None

    def _update_window_bbox(self, force: bool = False) -> bool:
        """Update cached window bounding box if needed.

        Args:
            force: Force update regardless of time elapsed.

        Returns:
            True if window bbox is valid, False otherwise.
        """
        current_time = time.time()

        if not force and (current_time - self._bbox_last_check) < self._bbox_check_interval:
            return self._window_bbox is not None

        self._bbox_last_check = current_time
        self._window_bbox = self.find_window()

        return self._window_bbox is not None

    def capture(self) -> NDArray[np.uint8] | None:
        """Capture the current game window frame.

        Returns:
            BGR numpy array of the captured frame, or None if capture failed.
        """
        if not self._update_window_bbox():
            return None

        bbox = self._window_bbox
        if bbox is None:
            return None

        return self.capture_region(bbox)

    def capture_region(self, region: tuple[int, int, int, int]) -> NDArray[np.uint8] | None:
        """Capture a specific region of the screen.

        Args:
            region: Tuple of (left, top, width, height) defining the capture area.

        Returns:
            BGR numpy array of the captured region, or None if capture failed.
        """
        left, top, width, height = region

        try:
            if self._use_dxcam_backend and self._dxcam_camera is not None:
                return self._capture_dxcam(left, top, width, height)
            else:
                return self._capture_mss(left, top, width, height)
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None

    def _capture_mss(
        self, left: int, top: int, width: int, height: int
    ) -> NDArray[np.uint8] | None:
        """Capture using mss backend.

        Args:
            left: Left coordinate of capture region.
            top: Top coordinate of capture region.
            width: Width of capture region.
            height: Height of capture region.

        Returns:
            BGR numpy array or None if capture failed.
        """
        if self._mss_instance is None:
            self._mss_instance = mss.mss()

        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        try:
            screenshot = self._mss_instance.grab(monitor)
            # mss returns BGRA, convert to BGR
            frame = np.array(screenshot, dtype=np.uint8)
            # Drop alpha channel: BGRA -> BGR
            frame = frame[:, :, :3]
            return frame
        except Exception as e:
            logger.debug(f"mss capture error: {e}")
            return None

    def _capture_dxcam(
        self, left: int, top: int, width: int, height: int
    ) -> NDArray[np.uint8] | None:
        """Capture using dxcam backend (Windows only).

        Args:
            left: Left coordinate of capture region.
            top: Top coordinate of capture region.
            width: Width of capture region.
            height: Height of capture region.

        Returns:
            BGR numpy array or None if capture failed.
        """
        if self._dxcam_camera is None:
            return None

        try:
            # dxcam expects region as (left, top, right, bottom)
            region = (left, top, left + width, top + height)
            frame = self._dxcam_camera.grab(region=region)
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
            return None
        except Exception as e:
            logger.debug(f"dxcam capture error: {e}")
            return None

    def start(self) -> None:
        """Start continuous background capture.

        Spawns a background thread that continuously captures frames at the
        target FPS and stores them in a buffer.
        """
        if self._running:
            logger.warning("Capture already running")
            return

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenCaptureThread",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info(f"Started continuous capture at {self.target_fps} FPS")

    def stop(self) -> None:
        """Stop continuous background capture."""
        if not self._running:
            return

        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
            self._capture_thread = None

        logger.info("Stopped continuous capture")

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        next_capture_time = time.time()

        while self._running:
            current_time = time.time()

            # Wait until next frame time
            if current_time < next_capture_time:
                sleep_time = next_capture_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Capture frame
            frame = self.capture()

            if frame is not None:
                with self._lock:
                    self._frame_buffer.append(frame)
                    self._latest_frame = frame

            # Schedule next capture
            next_capture_time = current_time + self._frame_interval

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the most recently captured frame.

        Thread-safe method to retrieve the latest frame from the buffer.

        Returns:
            BGR numpy array of the latest frame, or None if no frames captured.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_frame_buffer(self) -> list[NDArray[np.uint8]]:
        """Get all frames currently in the buffer.

        Useful for frame averaging or motion detection.

        Returns:
            List of BGR numpy arrays (oldest to newest).
        """
        with self._lock:
            return [frame.copy() for frame in self._frame_buffer]

    @property
    def is_running(self) -> bool:
        """Check if continuous capture is active.

        Returns:
            True if background capture is running.
        """
        return self._running

    @property
    def window_bbox(self) -> tuple[int, int, int, int] | None:
        """Get the current window bounding box.

        Returns:
            Tuple of (left, top, width, height) or None if window not found.
        """
        return self._window_bbox

    def __enter__(self) -> "ScreenCapture":
        """Context manager entry - starts capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops capture."""
        self.stop()

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        self.stop()

        if self._mss_instance is not None:
            try:
                self._mss_instance.close()
            except Exception:
                pass

        if self._dxcam_camera is not None:
            try:
                del self._dxcam_camera
            except Exception:
                pass
