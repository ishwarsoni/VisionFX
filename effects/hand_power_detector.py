"""
Hand Power Detector - Clean MediaPipe Hands implementation
"""

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from effects.simple_hand_tracker import SimpleHandData
from utils.logger import Logger


@dataclass
class HandData:
    """Hand tracking data."""

    palm_center: Tuple[float, float]
    palm_width: float
    fingers_extended: List[bool]
    hand_rotation: float
    velocity: Tuple[float, float]
    openness: float
    confidence: float
    handedness: str = "unknown"
    hand_scale: float = 0.0
    normalized_hand_size: float = 0.0


@dataclass
class PowerGesture:
    """Detected power gesture."""

    power_type: str
    confidence: float
    hand_data: HandData


class HandPowerDetector:
    """MediaPipe Hands detector with proper configuration."""

    def __init__(self):
        self._hands = None
        self._mediapipe_available = False
        self._last_palm = None
        self._last_detection = "Initializing..."
        self._frame_count = 0
        self._timestamp_ms = 0
        self._last_multi_results = []
        self.logger = Logger(name="HandPowerDetector", verbose=False)
        self._init_media_pipe()

    def _init_media_pipe(self):
        """Initialize MediaPipe Hands."""
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

            with self._suppress_external_logs():
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions, vision

            model_path = os.path.expanduser("~/.mediapipe/models/hand_landmarker.task")

            if not os.path.exists(model_path):
                self.logger.info("Downloading hand model...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                import urllib.request

                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
            with self._suppress_external_logs():
                self._hands = vision.HandLandmarker.create_from_options(options)
            self._mediapipe_available = True

        except Exception as e:
            self._mediapipe_available = False
            self._hands = None
            self._last_detection = f"No hand ({type(e).__name__})"

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom <= 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def process_frame(self, frame: np.ndarray):
        """Process frame and return first hand landmarks (legacy compatibility)."""
        results = self.process_hands(frame)
        if not results:
            return None
        return results[0]["landmarks"]

    def process_hands(self, frame: np.ndarray):
        """Process frame and return hand results with handedness metadata."""
        if not self._mediapipe_available:
            self._last_detection = "No hand"
            self._last_multi_results = []
            return None

        self._frame_count += 1
        self._timestamp_ms += 33

        try:
            with self._suppress_external_logs():
                import mediapipe as mp
                from mediapipe.tasks.python import vision

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect
            with self._suppress_external_logs():
                results = self._hands.detect_for_video(mp_image, self._timestamp_ms)

            if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
                packed = []
                for i, lms in enumerate(results.hand_landmarks):
                    handedness = "unknown"
                    score = 0.0
                    if getattr(results, "handedness", None) and i < len(
                        results.handedness
                    ):
                        cats = results.handedness[i]
                        if cats and len(cats) > 0:
                            category = cats[0]
                            handedness = str(
                                getattr(category, "category_name", "unknown")
                                or "unknown"
                            ).lower()
                            score = float(getattr(category, "score", 0.0) or 0.0)
                    packed.append(
                        {
                            "landmarks": lms,
                            "handedness": handedness,
                            "handedness_confidence": score,
                        }
                    )

                self._last_multi_results = packed
                labels = ",".join(item["handedness"] for item in packed)
                self._last_detection = f"HANDS: {len(packed)} [{labels}]"
                return packed

            self._last_detection = "No hand"
            self._last_multi_results = []
            return []

        except Exception as e:
            self._last_detection = f"No hand ({type(e).__name__})"
            self._last_multi_results = []
            return []

    def detect_hand(self, landmarks, frame_shape) -> Optional[HandData]:
        """Extract hand data from landmarks."""
        if not landmarks:
            return None

        if isinstance(landmarks, SimpleHandData):
            current_palm = landmarks.palm_center
            if self._last_palm:
                velocity = (
                    current_palm[0] - self._last_palm[0],
                    current_palm[1] - self._last_palm[1],
                )
            else:
                velocity = (0.0, 0.0)

            self._last_palm = current_palm

            return HandData(
                palm_center=landmarks.palm_center,
                palm_width=float(landmarks.palm_radius) * 2.0,
                fingers_extended=landmarks.fingers_extended,
                hand_rotation=landmarks.hand_rotation,
                velocity=velocity,
                openness=landmarks.openness,
                confidence=landmarks.confidence,
                hand_scale=float(landmarks.palm_radius) * 2.0,
                normalized_hand_size=float(landmarks.palm_radius)
                * 2.0
                / max(1.0, float(min(frame_shape))),
            )

        h, w = frame_shape

        # Get landmarks (21 points)
        def get_lm(idx):
            if hasattr(landmarks, "landmark"):
                return landmarks.landmark[idx]
            return landmarks[idx]

        points = np.array(
            [(float(get_lm(i).x) * w, float(get_lm(i).y) * h) for i in range(21)],
            dtype=np.float32,
        )

        # Palm center from MCP joints (5, 9, 13, 17)
        palm = np.mean(points[[5, 9, 13, 17]], axis=0)
        palm_x, palm_y = float(palm[0]), float(palm[1])

        # Distance-invariant hand scale from multiple cues.
        palm_width = float(np.linalg.norm(points[5] - points[17]))
        wrist_to_middle = float(np.linalg.norm(points[0] - points[12]))
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        hand_scale = max(
            1.0, float(np.mean([palm_width, wrist_to_middle * 0.8, bbox_diag * 0.6]))
        )

        # Finger extension by geometry normalized to hand scale.
        fingers = []

        thumb_span = float(np.linalg.norm(points[4] - points[2])) / hand_scale
        thumb_angle = self._angle(points[1], points[2], points[4])
        fingers.append(bool(thumb_span > 0.33 and thumb_angle > 130.0))

        for tip_idx, pip_idx, mcp_idx in (
            (8, 6, 5),
            (12, 10, 9),
            (16, 14, 13),
            (20, 18, 17),
        ):
            tip = points[tip_idx]
            pip = points[pip_idx]
            mcp = points[mcp_idx]
            wrist = points[0]

            tip_reach = float(np.linalg.norm(tip - wrist)) / hand_scale
            pip_reach = float(np.linalg.norm(pip - wrist)) / hand_scale
            finger_angle = self._angle(mcp, pip, tip)
            is_extended = tip_reach > (pip_reach + 0.07) and finger_angle > 145.0
            fingers.append(bool(is_extended))

        # Velocity
        if self._last_palm:
            vel_x = palm_x - self._last_palm[0]
            vel_y = palm_y - self._last_palm[1]
            velocity = (vel_x, vel_y)
        else:
            velocity = (0.0, 0.0)

        self._last_palm = (palm_x, palm_y)

        # Openness from normalized tip extension beyond MCP baseline.
        openness_components: List[float] = []
        for tip_idx, mcp_idx in ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17)):
            tip_dist = float(np.linalg.norm(points[tip_idx] - points[0])) / hand_scale
            mcp_dist = float(np.linalg.norm(points[mcp_idx] - points[0])) / hand_scale
            openness_components.append(
                float(np.clip((tip_dist - mcp_dist + 0.05) / 0.55, 0.0, 1.0))
            )
        openness = float(np.mean(openness_components)) if openness_components else 0.0
        normalized_hand_size = hand_scale / max(1.0, float(min(h, w)))

        return HandData(
            palm_center=(palm_x, palm_y),
            palm_width=palm_width,
            fingers_extended=fingers,
            hand_rotation=0,
            velocity=velocity,
            openness=openness,
            confidence=0.8,
            hand_scale=hand_scale,
            normalized_hand_size=normalized_hand_size,
        )

    def detect_hands(self, hand_results, frame_shape) -> List[HandData]:
        """Convert multi-hand MediaPipe output into HandData list with handedness."""
        if not hand_results:
            return []

        converted: List[HandData] = []
        for item in hand_results:
            if not isinstance(item, dict):
                continue
            lms = item.get("landmarks")
            if lms is None:
                continue
            hand = self.detect_hand(lms, frame_shape)
            if hand is None:
                continue
            handedness = str(item.get("handedness", "unknown")).lower()
            hand.handedness = handedness
            handedness_conf = float(item.get("handedness_confidence", 0.0) or 0.0)
            hand.confidence = max(hand.confidence, handedness_conf)
            converted.append(hand)

        return converted

    def detect_power_gesture(self, hand_data: HandData) -> Optional[PowerGesture]:
        """Detect power gesture."""
        if not hand_data:
            return None

        fingers = hand_data.fingers_extended
        open_count = sum(fingers)

        # Open palm (4+ fingers) = Rasengan
        if (
            open_count >= 4
            and hand_data.openness >= 0.75
            and hand_data.confidence >= 0.6
        ):
            return PowerGesture(
                power_type="rasengan", confidence=0.9, hand_data=hand_data
            )

        # Removed simple index-only Chidori trigger.
        # Chidori must be detected by higher-level classifier using landmarks.

        # Fist = deactivate
        if open_count == 0:
            return PowerGesture(
                power_type="tight_fist", confidence=0.95, hand_data=hand_data
            )

        return None

    def get_debug_info(self) -> str:
        """Get debug info."""
        return self._last_detection

    @staticmethod
    @contextmanager
    def _suppress_external_logs():
        with open(os.devnull, "w") as devnull:
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    yield
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)


def create_hand_detector() -> HandPowerDetector:
    return HandPowerDetector()
