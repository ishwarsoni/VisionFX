"""
Blink detection using eye aspect ratio analysis.
Detects eye blinks and tracks blink statistics.
"""

import time
from typing import List, Optional, Tuple

from tracking.tracking_data import BlinkData, EyeData
from utils.logger import Logger


class BlinkDetector:
    """
    Production-grade blink detection using eye aspect ratio and openness.
    Detects blinks and tracks duration and frequency.
    """

    # Default thresholds
    DEFAULT_BLINK_THRESHOLD = 0.3  # Eye openness below this = blink
    DEFAULT_BLINK_DURATION_MIN_MS = 50  # Minimum blink duration in ms
    DEFAULT_BLINK_DURATION_MAX_MS = 400  # Maximum blink duration in ms

    def __init__(
        self,
        blink_threshold: float = DEFAULT_BLINK_THRESHOLD,
        min_blink_duration_ms: float = DEFAULT_BLINK_DURATION_MIN_MS,
        max_blink_duration_ms: float = DEFAULT_BLINK_DURATION_MAX_MS,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize blink detector.

        Args:
            blink_threshold: Eye openness threshold for detecting blink
            min_blink_duration_ms: Minimum duration to count as blink
            max_blink_duration_ms: Maximum duration to count as blink
            logger: Logger instance
        """
        self.logger = logger or Logger(name="BlinkDetector")
        self.blink_threshold = max(0.0, min(1.0, blink_threshold))
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms

        # State tracking
        self.is_blinking = False
        self.blink_start_time: Optional[float] = None
        self.last_blink_time: Optional[float] = None
        self.blink_count = 0
        self.frame_count = 0

        self.logger.info("BlinkDetector initialized")

    def detect(self, left_eye: EyeData, right_eye: EyeData) -> BlinkData:
        """
        Detect blinks from both eyes.

        Args:
            left_eye: Left eye data
            right_eye: Right eye data

        Returns:
            BlinkData with blink information
        """
        self.frame_count += 1

        # Calculate average eye openness from both eyes
        avg_openness = (left_eye.openness + right_eye.openness) / 2.0

        # Detect blink state transition
        was_blinking = self.is_blinking
        self.is_blinking = avg_openness < self.blink_threshold

        # Calculate blink duration
        current_time = time.time()
        blink_duration_ms = 0.0

        # Blink started
        if self.is_blinking and not was_blinking:
            self.blink_start_time = current_time
            self.logger.debug(f"Blink started (frame {self.frame_count})")

        # Blink ongoing
        if self.is_blinking and self.blink_start_time:
            blink_duration_ms = (current_time - self.blink_start_time) * 1000.0

        # Blink ended
        if not self.is_blinking and was_blinking:
            if self.blink_start_time:
                blink_duration_ms = (current_time - self.blink_start_time) * 1000.0

                # Count as valid blink if within duration range
                if (
                    self.min_blink_duration_ms
                    <= blink_duration_ms
                    <= self.max_blink_duration_ms
                ):
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    self.logger.debug(
                        f"Blink detected (duration: {blink_duration_ms:.1f}ms, "
                        f"count: {self.blink_count})"
                    )

            self.blink_start_time = None

        # Create blink data
        blink_data = BlinkData(
            is_blinking=self.is_blinking,
            blink_start_frame=self.frame_count if self.is_blinking else None,
            blink_duration_ms=blink_duration_ms,
            eye_openness=avg_openness,
            blink_count=self.blink_count,
            last_blink_frame=self.frame_count if self.last_blink_time else None,
        )

        return blink_data

    def set_threshold(self, threshold: float) -> None:
        """
        Update blink threshold at runtime.

        Args:
            threshold: New threshold (0.0-1.0)
        """
        self.blink_threshold = max(0.0, min(1.0, threshold))
        self.logger.debug(f"Blink threshold updated to {self.blink_threshold:.3f}")

    def get_blink_frequency(self) -> float:
        """
        Get blink frequency (blinks per minute).

        Returns:
            Blinks per minute (approximate)
        """
        if self.frame_count == 0:
            return 0.0

        # Estimate based on frame count (assumes 30 FPS)
        estimated_seconds = self.frame_count / 30.0
        if estimated_seconds > 0:
            bpm = (self.blink_count / estimated_seconds) * 60.0
            return bpm

        return 0.0

    def reset(self) -> None:
        """Reset blink detector state."""
        self.is_blinking = False
        self.blink_start_time = None
        self.last_blink_time = None
        self.blink_count = 0
        self.frame_count = 0
        self.logger.debug("BlinkDetector reset")

    def get_stats(self) -> dict:
        """
        Get blink statistics.

        Returns:
            Dictionary with blink statistics
        """
        return {
            "blink_count": self.blink_count,
            "is_blinking": self.is_blinking,
            "blink_frequency_bpm": self.get_blink_frequency(),
            "frame_count": self.frame_count,
            "threshold": self.blink_threshold,
        }
