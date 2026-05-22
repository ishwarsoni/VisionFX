"""
Iris tracking and gaze direction estimation.
Detects iris center and estimates gaze direction.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.tracking_data import EyeData, GazeDirection, IrisData
from utils.logger import Logger


class IrisTracker:
    """
    Production-grade iris tracking and gaze estimation.
    Detects iris center and calculates gaze direction.
    """

    # MediaPipe iris landmark indices (in eye region)
    # These are relative to the eye region, not the full face
    IRIS_INDICES = [468, 469, 470, 471]  # Iris in MediaPipe full face mesh

    # Iris radius approximation in eye region
    IRIS_RADIUS_RATIO = 0.15  # Iris is ~15% of eye region width

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize iris tracker.

        Args:
            logger: Logger instance
        """
        self.logger = logger or Logger(name="IrisTracker")
        self.logger.info("IrisTracker initialized")

    def detect_iris(
        self,
        eye_landmarks: List[Tuple[float, float]],
        frame: np.ndarray,
        eye_side: str = "left",
    ) -> IrisData:
        """
        Detect iris center and gaze direction.

        Args:
            eye_landmarks: Eye region landmarks
            frame: Full frame for context (optional)
            eye_side: "left" or "right"

        Returns:
            IrisData with iris position and gaze direction
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return IrisData(center=(0.5, 0.5), radius=0.1, confidence=0.0)

        try:
            # Calculate eye bounding box
            xs = [lm[0] for lm in eye_landmarks]
            ys = [lm[1] for lm in eye_landmarks]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            eye_width = x_max - x_min
            eye_height = y_max - y_min

            if eye_width <= 0 or eye_height <= 0:
                return IrisData(center=(0.5, 0.5), radius=0.1, confidence=0.0)

            # Extract eye region
            x_min_int = max(0, int(x_min))
            y_min_int = max(0, int(y_min))
            x_max_int = min(frame.shape[1], int(x_max) + 1)
            y_max_int = min(frame.shape[0], int(y_max) + 1)

            eye_region = frame[y_min_int:y_max_int, x_min_int:x_max_int]

            if eye_region.size == 0:
                return IrisData(center=(0.5, 0.5), radius=0.1, confidence=0.0)

            # Detect iris using color analysis
            iris_center = self._detect_iris_center(eye_region)

            # Normalize to [0, 1] within eye region
            norm_x = iris_center[0] / eye_width if eye_width > 0 else 0.5
            norm_y = iris_center[1] / eye_height if eye_height > 0 else 0.5

            # Clamp to valid range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            # Estimate gaze direction from iris position
            gaze_direction = self._estimate_gaze_direction(norm_x, norm_y)

            # Estimate confidence based on iris visibility
            confidence = self._estimate_confidence(eye_region)

            iris_radius = self.IRIS_RADIUS_RATIO

            iris_data = IrisData(
                center=(norm_x, norm_y),
                radius=iris_radius,
                gaze_direction=gaze_direction,
                confidence=confidence,
            )

            return iris_data

        except Exception as e:
            self.logger.error(f"Iris detection error: {e}")
            return IrisData(center=(0.5, 0.5), radius=0.1, confidence=0.0)

    def _detect_iris_center(self, eye_region: np.ndarray) -> Tuple[float, float]:
        """
        Detect iris center in eye region using color analysis.

        Args:
            eye_region: Extracted eye region

        Returns:
            (x, y) coordinates of iris center
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Detect dark regions (iris)
            _, thresh = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Invert for iris detection
            thresh = cv2.bitwise_not(thresh)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return (eye_region.shape[1] / 2, eye_region.shape[0] / 2)

            # Find largest contour (likely the iris)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)
            else:
                return (eye_region.shape[1] / 2, eye_region.shape[0] / 2)

        except Exception as e:
            self.logger.debug(f"Iris center detection error: {e}")
            return (eye_region.shape[1] / 2, eye_region.shape[0] / 2)

    def _estimate_gaze_direction(self, norm_x: float, norm_y: float) -> GazeDirection:
        """
        Estimate gaze direction from normalized iris position.

        Args:
            norm_x: Normalized X position (0.0-1.0, 0.5=center)
            norm_y: Normalized Y position (0.0-1.0, 0.5=center)

        Returns:
            GazeDirection enum
        """
        # Define thresholds for direction detection
        HORIZONTAL_THRESHOLD = 0.35
        VERTICAL_THRESHOLD = 0.35
        CORNER_THRESHOLD = 0.25

        # Horizontal component
        is_left = norm_x < (0.5 - HORIZONTAL_THRESHOLD)
        is_right = norm_x > (0.5 + HORIZONTAL_THRESHOLD)
        is_center_h = not (is_left or is_right)

        # Vertical component
        is_up = norm_y < (0.5 - VERTICAL_THRESHOLD)
        is_down = norm_y > (0.5 + VERTICAL_THRESHOLD)
        is_center_v = not (is_up or is_down)

        # Determine direction
        if is_center_h and is_center_v:
            return GazeDirection.CENTER
        elif is_left and is_center_v:
            return GazeDirection.LEFT
        elif is_right and is_center_v:
            return GazeDirection.RIGHT
        elif is_center_h and is_up:
            return GazeDirection.UP
        elif is_center_h and is_down:
            return GazeDirection.DOWN
        elif is_left and is_up:
            return GazeDirection.UP_LEFT
        elif is_right and is_up:
            return GazeDirection.UP_RIGHT
        elif is_left and is_down:
            return GazeDirection.DOWN_LEFT
        elif is_right and is_down:
            return GazeDirection.DOWN_RIGHT
        else:
            return GazeDirection.CENTER

    def _estimate_confidence(self, eye_region: np.ndarray) -> float:
        """
        Estimate iris detection confidence based on image quality.

        Args:
            eye_region: Eye region image

        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            # Calculate image clarity using Laplacian variance
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize to 0.0-1.0 range
            # High variance = clear image = high confidence
            confidence = min(1.0, max(0.0, laplacian_var / 100.0))

            return confidence

        except:
            return 0.5

    def calculate_eye_openness(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate eye openness using aspect ratio.

        Args:
            eye_landmarks: Eye region landmarks

        Returns:
            Eye openness score (0.0=closed, 1.0=fully open)
        """
        if len(eye_landmarks) < 6:
            return 0.5

        try:
            # Use vertical distance at specific points
            # Points 2 and 4 are typically top and bottom of eye
            if len(eye_landmarks) >= 5:
                top = eye_landmarks[1][1]  # Upper eyelid
                bottom = eye_landmarks[4][1]  # Lower eyelid
                left = eye_landmarks[0][0]  # Left corner
                right = eye_landmarks[3][0]  # Right corner

                vertical_dist = abs(bottom - top)
                horizontal_dist = abs(right - left)

                if horizontal_dist > 0:
                    aspect_ratio = vertical_dist / horizontal_dist
                    # Normalize: typical ratio ranges 0.1-0.5
                    # 0.1 = closed, 0.5+ = open
                    openness = min(1.0, max(0.0, (aspect_ratio - 0.1) / 0.4))
                    return openness

        except Exception as e:
            self.logger.debug(f"Eye openness calculation error: {e}")

        return 0.5
