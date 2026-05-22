"""
Window management and display module.
"""

from typing import Optional

import cv2
import numpy as np

from utils.logger import Logger


class WindowManager:
    """
    Production-grade window management system.
    Handles window creation, fullscreen toggling, and display rendering.
    """

    def __init__(
        self,
        window_name: str = "Camera",
        width: int = 1280,
        height: int = 720,
        resizable: bool = True,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize window manager.

        Args:
            window_name: Name of the OpenCV window
            width: Initial window width
            height: Initial window height
            resizable: Whether window is resizable
            logger: Logger instance
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.resizable = resizable
        self.logger = logger or Logger(name="WindowManager")

        self.is_fullscreen = False
        self.is_created = False

        self._create_window()

    def _create_window(self) -> None:
        """Create and configure OpenCV window."""
        try:
            flags = cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE
            cv2.namedWindow(self.window_name, flags)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            self.is_created = True
            self.logger.info(
                f"Window created: {self.window_name} ({self.width}x{self.height})"
            )
        except Exception as e:
            self.logger.error(f"Failed to create window: {e}")
            self.is_created = False

    def display_frame(self, frame: np.ndarray) -> None:
        """
        Display frame in window.

        Args:
            frame: Image frame to display
        """
        if not self.is_created:
            return

        try:
            cv2.imshow(self.window_name, frame)
        except Exception as e:
            self.logger.error(f"Failed to display frame: {e}")

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if not self.is_created:
            return

        try:
            if self.is_fullscreen:
                # Exit fullscreen
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                )
                self.is_fullscreen = False
                self.logger.info("Fullscreen mode disabled")
            else:
                # Enter fullscreen
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
                self.is_fullscreen = True
                self.logger.info("Fullscreen mode enabled")
        except Exception as e:
            self.logger.error(f"Fullscreen toggle failed: {e}")

    def update_title(self, new_title: str) -> None:
        """
        Update window title dynamically.

        Args:
            new_title: New window title
        """
        if not self.is_created:
            return

        try:
            cv2.setWindowTitle(self.window_name, new_title)
            self.window_name = new_title
        except Exception as e:
            self.logger.error(f"Failed to update window title: {e}")

    def is_window_open(self) -> bool:
        """Check if window is still open."""
        if not self.is_created:
            return False

        try:
            # Try to get window properties to verify it's still open
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            return True
        except:
            return False

    def close(self) -> None:
        """Close window."""
        if self.is_created:
            try:
                cv2.destroyWindow(self.window_name)
                self.is_created = False
                self.logger.info("Window closed")
            except Exception as e:
                self.logger.error(f"Failed to close window: {e}")

    def __del__(self):
        """Ensure window is closed on object destruction."""
        self.close()
