"""
Simple OpenCV-based hand tracker fallback.
Uses improved skin detection + contour analysis + motion detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SimpleHandData:
    """Simple hand tracking data."""

    palm_center: Tuple[float, float]
    palm_radius: float
    fingers_extended: List[bool]
    hand_rotation: float
    openness: float
    confidence: float
    fingers_spread: float = 0.0


class SimpleHandTracker:
    MIN_AREA = 800
    MAX_AREA = 150000

    def __init__(self):
        self._last_palm = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, history=5
        )
        self._frame_count = 0
        self._last_contour_area = 0
        self._consecutive_valid = 0
        self._prev_frame = None
        self._motion_mask = None

    def process_frame(self, frame):
        """Process frame and return hand detection result with improved detection."""
        self._frame_count += 1

        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_frame is not None:
            diff = cv2.absdiff(gray, self._prev_frame)
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
        else:
            motion_mask = np.zeros((h, w), dtype=np.uint8)

        self._prev_frame = gray

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        lower_ycrcb = np.array((0, 133, 77), dtype=np.uint8)
        upper_ycrcb = np.array((255, 180, 133), dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 15, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(motion_mask // 4))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._consecutive_valid = 0
            return None

        valid_contours = [
            c for c in contours if self.MIN_AREA <= cv2.contourArea(c) <= self.MAX_AREA
        ]

        if not valid_contours:
            self._consecutive_valid = 0
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.MIN_AREA or area > self.MAX_AREA:
            self._consecutive_valid = 0
            return None

        try:
            ellipse = cv2.fitEllipse(largest)
            (cx, cy), (major, minor), angle = ellipse

            if cx < 30 or cx > w - 30 or cy < 30 or cy > h - 30:
                self._consecutive_valid = 0
                return None

            self._consecutive_valid += 1

            ratio = major / minor if minor > 0 else 1.0

            if ratio > 2.5:
                fingers_extended = [False, False, False, False, False]
                openness = 0.0
                confidence = 0.4
            elif ratio > 1.8:
                fingers_extended = [False, True, False, False, False]
                openness = 0.2
                confidence = 0.5
            elif ratio > 1.3:
                fingers_extended = [True, True, True, True, True]
                openness = 0.6
                confidence = 0.7
            else:
                fingers_extended = [True, True, True, True, True]
                openness = 0.8
                confidence = 0.8

            if self._consecutive_valid > 3:
                confidence = min(0.9, confidence + 0.1)

            return SimpleHandData(
                palm_center=(float(cx), float(cy)),
                palm_radius=float((major + minor) / 4),
                fingers_extended=fingers_extended,
                hand_rotation=angle,
                openness=openness,
                confidence=confidence,
                fingers_spread=0.5 if openness > 0.3 else 0.0,
            )
        except:
            self._consecutive_valid = 0
            return None

    def detect_hand(self, hand_result, frame_shape) -> Optional[SimpleHandData]:
        """Convert result to hand data."""
        if hand_result is None:
            return None
        return hand_result


def create_simple_hand_tracker() -> SimpleHandTracker:
    """Factory function."""
    return SimpleHandTracker()
