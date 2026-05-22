"""
Eye contact and gaze attention detection.
Determines if subject is looking at camera with sustained focus.
"""

import time
from typing import Optional

from tracking.tracking_data import EyeContactData, EyeData, TrackingState
from utils.logger import Logger


class EyeContactDetector:
    """
    Production-grade eye contact detection.
    Determines sustained direct camera gaze based on iris position and symmetry.
    """

    # Default thresholds
    DEFAULT_GAZE_CENTER_RATIO_THRESHOLD = 0.6  # How centered iris must be
    DEFAULT_SYMMETRY_THRESHOLD = 0.7  # Left-right symmetry requirement
    DEFAULT_DURATION_THRESHOLD_MS = 200.0  # Minimum duration for eye contact

    def __init__(
        self,
        gaze_center_threshold: float = DEFAULT_GAZE_CENTER_RATIO_THRESHOLD,
        symmetry_threshold: float = DEFAULT_SYMMETRY_THRESHOLD,
        duration_threshold_ms: float = DEFAULT_DURATION_THRESHOLD_MS,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize eye contact detector.

        Args:
            gaze_center_threshold: How centered iris must be (0.0-1.0)
            symmetry_threshold: Left-right iris symmetry requirement (0.0-1.0)
            duration_threshold_ms: Minimum sustained duration
            logger: Logger instance
        """
        self.logger = logger or Logger(name="EyeContactDetector")
        self.gaze_center_threshold = max(0.0, min(1.0, gaze_center_threshold))
        self.symmetry_threshold = max(0.0, min(1.0, symmetry_threshold))
        self.duration_threshold_ms = duration_threshold_ms

        # State tracking
        self.contact_start_time: Optional[float] = None
        self.is_contact_active = False

        self.logger.info("EyeContactDetector initialized")

    def detect(
        self,
        left_eye: EyeData,
        right_eye: EyeData,
        left_openness: float,
        right_openness: float,
    ) -> EyeContactData:
        """
        Detect eye contact and sustained gaze.

        Args:
            left_eye: Left eye tracking data
            right_eye: Right eye tracking data
            left_openness: Left eye openness (0.0-1.0)
            right_openness: Right eye openness (0.0-1.0)

        Returns:
            EyeContactData with eye contact information
        """
        current_time = time.time()

        # Check if both eyes are open enough
        min_openness = 0.3  # Minimum openness for eye contact
        eyes_open = (left_openness > min_openness) and (right_openness > min_openness)

        if not eyes_open:
            self.contact_start_time = None
            self.is_contact_active = False
            return EyeContactData(in_contact=False, duration_ms=0.0)

        # Calculate iris position centrality
        left_centered = self._calculate_gaze_centered_ratio(left_eye.iris.center)
        right_centered = self._calculate_gaze_centered_ratio(right_eye.iris.center)

        # Check if irises are centered
        gaze_centered = (
            left_centered > self.gaze_center_threshold
            and right_centered > self.gaze_center_threshold
        )

        if not gaze_centered:
            self.contact_start_time = None
            self.is_contact_active = False
            return EyeContactData(in_contact=False, duration_ms=0.0)

        # Calculate iris symmetry
        iris_symmetry = self._calculate_iris_symmetry(
            left_eye.iris.center, right_eye.iris.center
        )

        # Check symmetry requirement
        symmetry_ok = iris_symmetry > self.symmetry_threshold

        if not symmetry_ok:
            self.contact_start_time = None
            self.is_contact_active = False
            return EyeContactData(
                in_contact=False,
                duration_ms=0.0,
                iris_symmetry=iris_symmetry,
                gaze_centered_ratio=(left_centered + right_centered) / 2.0,
            )

        # Track eye contact duration
        if self.contact_start_time is None:
            self.contact_start_time = current_time

        duration_ms = (current_time - self.contact_start_time) * 1000.0

        # Determine if eye contact is established
        is_contact = duration_ms >= self.duration_threshold_ms
        self.is_contact_active = is_contact

        if is_contact:
            self.logger.debug(f"Eye contact detected (duration: {duration_ms:.0f}ms)")

        # Calculate confidence
        confidence = min(1.0, duration_ms / 1000.0)  # Increase confidence over time

        eye_contact_data = EyeContactData(
            in_contact=is_contact,
            confidence=confidence,
            duration_ms=duration_ms,
            iris_symmetry=iris_symmetry,
            gaze_centered_ratio=(left_centered + right_centered) / 2.0,
        )

        return eye_contact_data

    def _calculate_gaze_centered_ratio(self, iris_position: tuple) -> float:
        """
        Calculate how centered the iris is (0.0=edge, 1.0=perfect center).

        Args:
            iris_position: (x, y) normalized iris position (0.0-1.0)

        Returns:
            Centered ratio (0.0-1.0)
        """
        x, y = iris_position

        # Distance from center (0.5, 0.5)
        dist_from_center = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5

        # Maximum possible distance (corner to center)
        max_distance = 0.5 * 2**0.5  # ~0.707

        # Normalized distance (0.0=center, 1.0=edge)
        norm_distance = min(1.0, dist_from_center / max_distance)

        # Invert to get "centered ratio"
        centered_ratio = 1.0 - norm_distance

        return centered_ratio

    def _calculate_iris_symmetry(self, left_iris: tuple, right_iris: tuple) -> float:
        """
        Calculate left-right iris symmetry.

        Args:
            left_iris: (x, y) normalized left iris position
            right_iris: (x, y) normalized right iris position

        Returns:
            Symmetry score (0.0-1.0, 1.0=perfectly symmetric)
        """
        # Calculate vertical distance difference (how level the irises are)
        y_diff = abs(left_iris[1] - right_iris[1])

        # Calculate horizontal position difference
        x_diff = abs(left_iris[0] - right_iris[0])

        # Combine metrics (lower is better)
        total_diff = y_diff + (x_diff * 0.5)  # Weight x difference less

        # Normalize to 0.0-1.0 (1.0=symmetric, 0.0=asymmetric)
        symmetry = max(0.0, 1.0 - total_diff)

        return symmetry

    def set_duration_threshold(self, threshold_ms: float) -> None:
        """
        Update duration threshold at runtime.

        Args:
            threshold_ms: New threshold in milliseconds
        """
        self.duration_threshold_ms = max(0.0, threshold_ms)
        self.logger.debug(f"Eye contact duration threshold updated to {threshold_ms}ms")

    def reset(self) -> None:
        """Reset eye contact detector state."""
        self.contact_start_time = None
        self.is_contact_active = False
        self.logger.debug("EyeContactDetector reset")

    def get_state(self) -> str:
        """Get current eye contact state."""
        if self.is_contact_active:
            return "EYE_CONTACT"
        elif self.contact_start_time is not None:
            return "APPROACHING_CONTACT"
        else:
            return "NO_CONTACT"
