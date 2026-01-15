"""Logging utilities for Auto-Balatro.

This module provides a configurable logging system with colored console output
and optional file logging support.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class ANSIColors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Log level colors
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta


def _supports_color() -> bool:
    """Check if the terminal supports color output.

    Returns:
        bool: True if terminal supports colors, False otherwise.
    """
    # Check if stdout is a terminal
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for NO_COLOR environment variable (standard convention)
    if os.environ.get("NO_COLOR"):
        return False

    # Check for TERM environment variable
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False

    # Windows check
    if sys.platform == "win32":
        # Windows 10+ supports ANSI codes in newer terminals
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences on Windows
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False

    return True


class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds color to log messages based on level.

    This formatter applies ANSI color codes to log messages when outputting
    to a terminal that supports colors.
    """

    LEVEL_COLORS = {
        logging.DEBUG: ANSIColors.DEBUG,
        logging.INFO: ANSIColors.INFO,
        logging.WARNING: ANSIColors.WARNING,
        logging.ERROR: ANSIColors.ERROR,
        logging.CRITICAL: ANSIColors.CRITICAL,
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        """Initialize the colored formatter.

        Args:
            fmt: The log message format string.
            datefmt: The date format string.
            use_colors: Whether to apply colors to output.
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional color.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted log message.
        """
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, ANSIColors.RESET)
            # Color the level name
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{ANSIColors.RESET}"
            formatted = super().format(record)
            record.levelname = original_levelname
            return formatted
        return super().format(record)


# Module-level flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure the root logger with console and optional file handlers.

    Sets up logging with colored console output and optional file logging.
    This function should typically be called once at application startup.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_file: Optional path to a log file. If provided, logs will also
            be written to this file. Parent directories will be created
            if they don't exist.

    Returns:
        logging.Logger: The configured root logger.

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """
    global _logging_configured

    # Get the root logger
    root_logger = logging.getLogger()

    # Clear existing handlers if reconfiguring
    if _logging_configured:
        root_logger.handlers.clear()

    # Set the logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Define format strings
    log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(
        fmt=log_format,
        datefmt=date_format,
        use_colors=True,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        # Create parent directories if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _logging_configured = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a specific module.

    This function returns a logger instance that can be used for logging
    within a specific module or component. The logger inherits settings
    from the root logger configured by setup_logging().

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: A logger instance with the specified name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    return logging.getLogger(name)
