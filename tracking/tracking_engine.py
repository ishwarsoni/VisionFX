"""
Main tracking system orchestrator.
Coordinates face tracking, iris detection, blink detection, and eye contact.
"""

import time
from typing import Optional

import cv2
import numpy as np

from tracking.blink_detector import BlinkDetector
from tracking.eye_contact_detector import EyeContactDetector
from tracking.face_tracker import FaceTracker
from tracking.iris_tracker import IrisTracker
from tracking.smoothing import ConfidenceSmoother, LandmarkSmoother, MultiPointSmoother
from tracking.tracking_data import (
    BlinkData,
    EyeContactData,
    EyeData,
    FaceData,
    IrisData,
    TrackingFrameData,
    TrackingState,
)
from utils.logger import Logger


class TrackingEngine:
    """
    Production-grade tracking engine orchestrating all subsystems.
    Provides unified interface for face, iris, blink, and eye contact tracking.
    """

    def __init__(self, smoothing_factor: float = 0.7, logger: Optional[Logger] = None):
        """
        Initialize tracking engine.

        Args:
            smoothing_factor: Smoothing factor for landmarks (0.0-1.0)
            logger: Logger instance
        """
        self.logger = logger or Logger(name="TrackingEngine")
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))

        # Initialize subsystems
        try:
            self.face_tracker = FaceTracker(logger=self.logger)
            self.iris_tracker = IrisTracker(logger=self.logger)
            self.blink_detector = BlinkDetector(logger=self.logger)
            self.eye_contact_detector = EyeContactDetector(logger=self.logger)

            self.landmark_smoother = LandmarkSmoother(
                smoothing_factor=smoothing_factor * 0.6
            )
            self.iris_smoother = MultiPointSmoother(
                smoothing_factor=smoothing_factor * 0.5
            )
            self.confidence_smoother = ConfidenceSmoother(smoothing_factor=0.6)

            self.is_initialized = True
            self.logger.info("TrackingEngine initialized successfully")

        except Exception as e:
            self.logger.error(f"TrackingEngine initialization failed: {e}")
            self.is_initialized = False
            raise

        # State
        self.frame_count = 0
        self.last_tracking_data: Optional[TrackingFrameData] = None

    def process_frame(self, frame: np.ndarray) -> TrackingFrameData:
        """
        Process frame and perform all tracking operations.

        Args:
            frame: Input BGR frame

        Returns:
            TrackingFrameData with all tracking information
        """
        self.frame_count += 1

        if not self.is_initialized:
            self.logger.error("TrackingEngine not initialized")
            return TrackingFrameData(frame_index=self.frame_count)

        try:
            # Create tracking data container
            tracking_data = TrackingFrameData(frame_index=self.frame_count)
            tracking_data.timestamp_ms = time.perf_counter() * 1000.0

            # 1. Detect face
            face_data = self.face_tracker.detect(frame)
            if not face_data:
                face_data = FaceData(detected=False)

            tracking_data.face = face_data

            # If no face detected, return early
            if not face_data.detected:
                tracking_data.state = TrackingState.NO_FACE
                tracking_data.overall_confidence = 0.0
                self.last_tracking_data = tracking_data
                return tracking_data

            # Smooth face landmarks
            smoothed_landmarks = self.landmark_smoother.smooth(face_data.landmarks)
            tracking_data.face.landmarks = smoothed_landmarks

            # 2. Detect irises for both eyes
            left_eye_landmarks = self.face_tracker.get_eye_region_landmarks(
                face_data, "left"
            )
            right_eye_landmarks = self.face_tracker.get_eye_region_landmarks(
                face_data, "right"
            )

            # Get eye openness
            left_openness = self.iris_tracker.calculate_eye_openness(left_eye_landmarks)
            right_openness = self.iris_tracker.calculate_eye_openness(
                right_eye_landmarks
            )

            # Detect irises
            left_iris_data = self.iris_tracker.detect_iris(
                left_eye_landmarks, frame, "left"
            )
            right_iris_data = self.iris_tracker.detect_iris(
                right_eye_landmarks, frame, "right"
            )

            # Smooth iris positions
            left_iris_smoothed = self.iris_smoother.smooth(
                "left_iris", left_iris_data.center
            )
            right_iris_smoothed = self.iris_smoother.smooth(
                "right_iris", right_iris_data.center
            )

            left_iris_data.center = left_iris_smoothed
            right_iris_data.center = right_iris_smoothed

            # Create eye data
            left_eye = EyeData(
                iris=left_iris_data,
                landmarks=left_eye_landmarks,
                openness=left_openness,
                aspect_ratio=left_openness,
            )

            right_eye = EyeData(
                iris=right_iris_data,
                landmarks=right_eye_landmarks,
                openness=right_openness,
                aspect_ratio=right_openness,
            )

            tracking_data.left_eye = left_eye
            tracking_data.right_eye = right_eye

            # 3. Detect blinks
            blink_data = self.blink_detector.detect(left_eye, right_eye)
            tracking_data.blink = blink_data

            # 4. Detect eye contact
            eye_contact_data = self.eye_contact_detector.detect(
                left_eye, right_eye, left_openness, right_openness
            )

            tracking_data.eye_contact = eye_contact_data

            # 5. Determine overall tracking state
            tracking_data.state = self._determine_tracking_state(
                face_data, blink_data, eye_contact_data
            )

            # 6. Calculate overall confidence
            face_conf = face_data.confidence
            iris_conf = (left_iris_data.confidence + right_iris_data.confidence) / 2.0
            overall_conf = (face_conf + iris_conf) / 2.0

            # Smooth confidence
            smoothed_conf = self.confidence_smoother.smooth(overall_conf)
            tracking_data.overall_confidence = smoothed_conf

            self.last_tracking_data = tracking_data
            return tracking_data

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return TrackingFrameData(frame_index=self.frame_count)

    def _determine_tracking_state(
        self,
        face_data: FaceData,
        blink_data: BlinkData,
        eye_contact_data: EyeContactData,
    ) -> TrackingState:
        """
        Determine overall tracking state based on subsystem states.

        Args:
            face_data: Face detection data
            blink_data: Blink detection data
            eye_contact_data: Eye contact data

        Returns:
            TrackingState enum
        """
        if not face_data.detected:
            return TrackingState.NO_FACE

        if blink_data.is_blinking:
            return TrackingState.BLINKING

        if eye_contact_data.in_contact:
            return TrackingState.EYE_CONTACT

        return TrackingState.TRACKING

    def update_config(self, **kwargs) -> None:
        """
        Update tracking configuration at runtime.

        Supported kwargs:
        - smoothing_factor: Update smoothing for all smoothers
        - blink_threshold: Update blink detection threshold
        - eye_contact_duration_ms: Update eye contact duration requirement
        """
        if "smoothing_factor" in kwargs:
            factor = kwargs["smoothing_factor"]
            self.smoothing_factor = max(0.0, min(1.0, factor))
            self.landmark_smoother.set_smoothing_factor(factor)
            self.iris_smoother.set_smoothing_factor(factor)
            self.logger.debug(f"Smoothing factor updated to {factor}")

        if "blink_threshold" in kwargs:
            threshold = kwargs["blink_threshold"]
            self.blink_detector.set_threshold(threshold)

        if "eye_contact_duration_ms" in kwargs:
            duration = kwargs["eye_contact_duration_ms"]
            self.eye_contact_detector.set_duration_threshold(duration)

    def reset(self) -> None:
        """Reset tracking engine state."""
        self.landmark_smoother.reset()
        self.iris_smoother.reset()
        self.confidence_smoother.reset()
        self.blink_detector.reset()
        self.eye_contact_detector.reset()
        self.frame_count = 0
        self.last_tracking_data = None
        self.logger.info("TrackingEngine reset")

    def get_last_data(self) -> Optional[TrackingFrameData]:
        """Get last tracking data (without reprocessing)."""
        return self.last_tracking_data

    def release(self) -> None:
        """Release resources."""
        if self.face_tracker:
            self.face_tracker.release()
        self.logger.info("TrackingEngine released")

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.release()
