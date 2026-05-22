"""
Landmark smoothing and temporal stabilization.
Ensures smooth visual tracking without jitter.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class LandmarkSmoother:
    """
    Production-grade landmark smoothing using exponential moving average.
    Reduces jitter while minimizing latency.
    """

    def __init__(self, smoothing_factor: float = 0.7, window_size: int = 5):
        """
        Initialize smoother.

        Args:
            smoothing_factor: EMA factor (0.0-1.0). Higher = more smoothing but higher latency
            window_size: Size of history window for fallback averaging
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.window_size = max(1, window_size)

        self.smoothed_landmarks: Optional[np.ndarray] = None
        self.landmark_history: deque = deque(maxlen=window_size)
        self.is_initialized = False

    def smooth(self, landmarks: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Apply smoothing to landmark sequence.

        Args:
            landmarks: List of (x, y) landmark positions

        Returns:
            Smoothed landmarks
        """
        current = np.array(landmarks, dtype=np.float32)

        # Initialize on first frame
        if not self.is_initialized:
            self.smoothed_landmarks = current.copy()
            self.is_initialized = True
            self.landmark_history.append(current)
            return landmarks

        # Apply exponential moving average
        self.smoothed_landmarks = (
            self.smoothing_factor * current
            + (1.0 - self.smoothing_factor) * self.smoothed_landmarks
        )

        self.landmark_history.append(self.smoothed_landmarks.copy())

        # Convert back to list of tuples
        return [(float(p[0]), float(p[1])) for p in self.smoothed_landmarks]

    def smooth_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Smooth a single point.

        Args:
            point: (x, y) coordinate

        Returns:
            Smoothed coordinate
        """
        smoothed = self.smooth([point])
        return smoothed[0]

    def reset(self) -> None:
        """Reset smoothing state."""
        self.smoothed_landmarks = None
        self.landmark_history.clear()
        self.is_initialized = False

    def set_smoothing_factor(self, factor: float) -> None:
        """
        Update smoothing factor at runtime.

        Args:
            factor: New smoothing factor (0.0-1.0)
        """
        self.smoothing_factor = max(0.0, min(1.0, factor))


class MultiPointSmoother:
    """
    Smooth multiple independent points with separate trackers.
    Useful for tracking iris, landmarks, etc.
    """

    def __init__(self, smoothing_factor: float = 0.7):
        """
        Initialize multi-point smoother.

        Args:
            smoothing_factor: Smoothing factor for all points
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.smoothers: Dict[str, LandmarkSmoother] = {}

    def smooth(self, label: str, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Smooth a named point.

        Args:
            label: Point identifier (e.g., "left_iris", "right_iris")
            point: (x, y) coordinate

        Returns:
            Smoothed coordinate
        """
        if label not in self.smoothers:
            self.smoothers[label] = LandmarkSmoother(self.smoothing_factor)

        return self.smoothers[label].smooth_point(point)

    def reset(self, label: Optional[str] = None) -> None:
        """
        Reset smoothing state.

        Args:
            label: Specific label to reset, or None to reset all
        """
        if label is None:
            self.smoothers.clear()
        elif label in self.smoothers:
            self.smoothers[label].reset()

    def set_smoothing_factor(self, factor: float) -> None:
        """Update smoothing factor for all points."""
        self.smoothing_factor = max(0.0, min(1.0, factor))
        for smoother in self.smoothers.values():
            smoother.set_smoothing_factor(factor)


class ConfidenceSmoother:
    """
    Smooth confidence scores to avoid flickering.
    Uses exponential moving average.
    """

    def __init__(self, smoothing_factor: float = 0.8):
        """
        Initialize confidence smoother.

        Args:
            smoothing_factor: How aggressively to smooth (0.0-1.0)
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.smoothed_value = 0.0
        self.is_initialized = False

    def smooth(self, value: float) -> float:
        """
        Smooth a confidence value.

        Args:
            value: Raw confidence value (0.0-1.0)

        Returns:
            Smoothed confidence value
        """
        value = max(0.0, min(1.0, value))

        if not self.is_initialized:
            self.smoothed_value = value
            self.is_initialized = True
            return value

        self.smoothed_value = (
            self.smoothing_factor * value
            + (1.0 - self.smoothing_factor) * self.smoothed_value
        )

        return self.smoothed_value

    def reset(self) -> None:
        """Reset smoother state."""
        self.smoothed_value = 0.0
        self.is_initialized = False
