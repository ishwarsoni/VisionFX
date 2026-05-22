"""
Logging system for consistent, professional console output.
"""

import sys
from datetime import datetime
from typing import Optional


class Logger:
    """
    Production-grade logger with colored console output.
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    def __init__(self, name: str = "App", verbose: bool = True):
        """
        Initialize logger.

        Args:
            name: Logger name (component identifier)
            verbose: Whether to show debug messages
        """
        self.name = name
        self.verbose = verbose

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _format_message(self, level: str, level_color: str, message: str) -> str:
        """Format log message with colors."""
        timestamp = self._timestamp()
        name_part = f"{self.CYAN}[{self.name}]{self.RESET}"
        level_part = f"{level_color}{level}{self.RESET}"

        return f"{self.GRAY}{timestamp}{self.RESET} {level_part} {name_part} {message}"

    def info(self, message: str) -> None:
        """Log info message."""
        formatted = self._format_message("INFO", f"{self.GREEN}{self.BOLD}", message)
        print(formatted, file=sys.stdout)

    def debug(self, message: str) -> None:
        """Log debug message."""
        if not self.verbose:
            return
        formatted = self._format_message("DEBUG", f"{self.WHITE}{self.DIM}", message)
        print(formatted, file=sys.stdout)

    def warning(self, message: str) -> None:
        """Log warning message."""
        formatted = self._format_message("WARN", f"{self.YELLOW}{self.BOLD}", message)
        print(formatted, file=sys.stderr)

    def error(self, message: str) -> None:
        """Log error message."""
        formatted = self._format_message("ERROR", f"{self.RED}{self.BOLD}", message)
        print(formatted, file=sys.stderr)

    def success(self, message: str) -> None:
        """Log success message."""
        formatted = self._format_message("✓", f"{self.GREEN}", message)
        print(formatted, file=sys.stdout)
