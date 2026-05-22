"""
Debug overlay rendering system.
Provides professional on-screen information display.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import Logger


class DebugOverlay:
    """
    Production-grade debug overlay system.
    Renders FPS, resolution, and status information with professional typography.
    """

    # Typography configuration
    FONT = cv2.FONT_HERSHEY_TRIPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    LINE_COLOR = (0, 255, 200)  # Cyan
    BG_COLOR = (20, 20, 40)  # Dark blue
    SHADOW_COLOR = (0, 0, 0)
    PADDING = 15
    LINE_HEIGHT = 28

    def __init__(self, enabled: bool = True, logger: Optional[Logger] = None):
        """
        Initialize debug overlay.

        Args:
            enabled: Whether overlay is initially enabled
            logger: Logger instance
        """
        self.enabled = enabled
        self.logger = logger or Logger(name="DebugOverlay")

        self.is_recording = False
        self.custom_info: List[Tuple[str, str]] = []

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        resolution: Tuple[int, int],
        recording: bool = False,
        active_states: Optional[dict] = None,
        custom_lines: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Render debug overlay on frame.

        Args:
            frame: Input frame
            fps: Current FPS value
            resolution: Frame resolution (width, height)
            recording: Whether recording is active
            active_states: Dictionary of active state indicators

        Returns:
            Frame with overlay rendered
        """
        if not self.enabled:
            return frame

        frame_copy = frame.copy()
        self.is_recording = recording

        # Build info lines
        info_lines = self._build_info_lines(
            fps, resolution, recording, active_states, custom_lines
        )

        # Render info panel
        self._render_info_panel(frame_copy, info_lines)

        return frame_copy

    def _build_info_lines(
        self,
        fps: float,
        resolution: Tuple[int, int],
        recording: bool,
        active_states: Optional[dict],
        custom_lines: Optional[List[str]],
    ) -> List[str]:
        """Build list of information lines to display."""
        lines = []

        # FPS
        lines.append(f"FPS: {fps:.1f}")

        # Resolution
        lines.append(f"RES: {resolution[0]}x{resolution[1]}")

        # Recording state
        if recording:
            lines.append("[X] RECORDING")

        # Active states
        if active_states:
            for state_name, is_active in active_states.items():
                status = "[X]" if is_active else "[ ]"
                lines.append(f"{status} {state_name.upper()}")

        if custom_lines:
            lines.append("")
            lines.extend(custom_lines)

        return lines

    def _render_info_panel(self, frame: np.ndarray, lines: List[str]) -> None:
        """
        Render semi-transparent info panel with text.

        Args:
            frame: Frame to render onto (modified in place)
            lines: List of text lines to display
        """
        if not lines:
            return

        h, w = frame.shape[:2]

        # Calculate panel dimensions
        panel_height = len(lines) * self.LINE_HEIGHT + self.PADDING * 2
        panel_width = 280

        # Top-left corner position
        x_offset = self.PADDING
        y_offset = self.PADDING

        # Create semi-transparent overlay for background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_offset, y_offset),
            (x_offset + panel_width, y_offset + panel_height),
            self.BG_COLOR,
            -1,
        )

        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + panel_width, y_offset + panel_height),
            self.LINE_COLOR,
            2,
        )

        # Render text lines
        for idx, line in enumerate(lines):
            y_pos = y_offset + self.PADDING + (idx * self.LINE_HEIGHT) + 20

            # Shadow effect
            cv2.putText(
                frame,
                line,
                (x_offset + self.PADDING + 1, y_pos + 1),
                self.FONT,
                self.FONT_SCALE,
                self.SHADOW_COLOR,
                self.FONT_THICKNESS,
            )

            # Main text
            cv2.putText(
                frame,
                line,
                (x_offset + self.PADDING, y_pos),
                self.FONT,
                self.FONT_SCALE,
                self.LINE_COLOR,
                self.FONT_THICKNESS,
            )

    def toggle(self) -> None:
        """Toggle overlay visibility."""
        self.enabled = not self.enabled
        self.logger.info(f"Debug overlay {'enabled' if self.enabled else 'disabled'}")

    def set_enabled(self, enabled: bool) -> None:
        """Set overlay enabled state."""
        self.enabled = enabled
