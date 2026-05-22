"""
AI-Driven Eye Segmentation and Compositing
Provides per-pixel eye awareness for realistic Sharingan transformation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EyeSegmentationMask:
    """Complete eye segmentation with multiple layers."""

    eye_region: np.ndarray
    sclera_mask: np.ndarray
    iris_mask: np.ndarray
    pupil_mask: np.ndarray
    upper_lid_mask: np.ndarray
    lower_lid_mask: np.ndarray
    skin_mask: np.ndarray
    eye_corner_left: Tuple[int, int]
    eye_corner_right: Tuple[int, int]
    confidence: float


@dataclass
class GazeState:
    """Gaze direction and iris position."""

    direction: str
    horizontal_angle: float
    vertical_angle: float
    iris_offset: Tuple[float, float]
    confidence: float


@dataclass
class EyeGeometry:
    """3D eye geometry for perspective-correct compositing."""

    eyeball_radius: float
    curvature_factor: float
    compression_x: float
    compression_y: float
    tilt_angle: float
    depth_scale: float


class AIEyeSegmenter:
    """
    AI-driven eye segmentation using MediaPipe Iris landmarks.
    Provides per-pixel eye awareness for realistic compositing.
    """

    LEFT_EYE_LANDMARKS = list(range(33, 54))
    RIGHT_EYE_LANDMARKS = list(range(263, 284))
    IRIS_LANDMARKS = list(range(468, 478))
    UPPER_LID_INDICES = [33, 34, 35, 36, 37, 38, 156, 159]
    LOWER_LID_INDICES = [41, 40, 39, 145, 144, 143, 153, 158]

    def __init__(self):
        self._last_masks = {}

    def segment_eye(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float]],
        eye_side: str = "left",
    ) -> Optional[EyeSegmentationMask]:
        """Generate complete eye segmentation mask."""
        if not landmarks or len(landmarks) < 478:
            return None

        side_idx = (
            self.LEFT_EYE_LANDMARKS if eye_side == "left" else self.RIGHT_EYE_LANDMARKS
        )

        eye_points = [landmarks[i] for i in side_idx if i < len(landmarks)]
        if len(eye_points) < 8:
            return None

        h, w = frame.shape[:2]

        eye_bb = self._get_eye_bounding_box(eye_points, w, h)
        x1, y1, x2, y2 = eye_bb

        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w < 10 or roi_h < 10:
            return None

        iris_center, iris_radius = self._get_iris_from_landmarks(landmarks, eye_side)

        sclera_mask = self._create_sclera_mask(roi_w, roi_h, iris_center, iris_radius)
        iris_mask = self._create_iris_mask(roi_w, roi_h, iris_center, iris_radius)
        pupil_mask = self._create_pupil_mask(
            roi_w, roi_h, iris_center, iris_radius * 0.6
        )

        upper_lid_pts = [
            landmarks[i] for i in self.UPPER_LID_INDICES if i < len(landmarks)
        ]
        lower_lid_pts = [
            landmarks[i] for i in self.LOWER_LID_INDICES if i < len(landmarks)
        ]

        upper_lid_mask = self._create_eyelid_mask(
            upper_lid_pts, (x1, y1), roi_w, roi_h, True
        )
        lower_lid_mask = self._create_eyelid_mask(
            lower_lid_pts, (x1, y1), roi_w, roi_h, False
        )

        eye_region = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(eye_region, [self._points_to_contour(eye_points, (x1, y1))], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_CLOSE, kernel)

        skin_mask = self._create_skin_mask(frame, eye_bb)

        outer_corner = (int(eye_points[0][0] - x1), int(eye_points[0][1] - y1))
        inner_corner = (int(eye_points[-1][0] - x1), int(eye_points[-1][1] - y1))

        confidence = self._compute_segmentation_confidence(
            eye_region, iris_mask, upper_lid_mask, lower_lid_mask
        )

        return EyeSegmentationMask(
            eye_region=eye_region,
            sclera_mask=sclera_mask,
            iris_mask=iris_mask,
            pupil_mask=pupil_mask,
            upper_lid_mask=upper_lid_mask,
            lower_lid_mask=lower_lid_mask,
            skin_mask=skin_mask,
            eye_corner_left=outer_corner,
            eye_corner_right=inner_corner,
            confidence=confidence,
        )

    def _get_eye_bounding_box(
        self,
        eye_points: List[Tuple[float, float]],
        img_w: int,
        img_h: int,
        padding: float = 0.4,
    ) -> Tuple[int, int, int, int]:
        """Calculate eye bounding box with padding."""
        xs = [p[0] for p in eye_points]
        ys = [p[1] for p in eye_points]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        w = x_max - x_min
        h = y_max - y_min

        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(img_w, x_max + pad_x)
        y2 = min(img_h, y_max + pad_y)

        return x1, y1, x2, y2

    def _get_iris_from_landmarks(
        self, landmarks: List[Tuple[float, float]], eye_side: str
    ) -> Tuple[Tuple[float, float], float]:
        """Extract iris center and radius from MediaPipe landmarks."""
        iris_idx = self.IRIS_INDICES

        iris_points = [landmarks[i] for i in iris_idx if i < len(landmarks)]

        cx = sum(p[0] for p in iris_points) / len(iris_points)
        cy = sum(p[1] for p in iris_points) / len(iris_points)

        max_dist = 0
        for p in iris_points:
            dist = np.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            max_dist = max(max_dist, dist)

        return (cx, cy), max_dist

    def _create_sclera_mask(
        self, w: int, h: int, iris_center: Tuple[float, float], iris_radius: float
    ) -> np.ndarray:
        """Create sclera (white of eye) mask using ellipse fit."""
        mask = np.zeros((h, w), dtype=np.float32)

        cx = iris_center[0]
        cy = iris_center[1]

        radius_x = iris_radius * 2.5
        radius_y = iris_radius * 1.8

        cv2.ellipse(
            mask, (int(cx), int(cy)), (int(radius_x), int(radius_y)), 0, 0, 360, 1.0, -1
        )

        inner_radius = iris_radius * 1.1
        cv2.circle(mask, (int(cx), int(cy)), int(inner_radius), 0.0, -1)

        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask

    def _create_iris_mask(
        self, w: int, h: int, iris_center: Tuple[float, float], iris_radius: float
    ) -> np.ndarray:
        """Create iris segmentation mask."""
        mask = np.zeros((h, w), dtype=np.float32)

        cx, cy = int(iris_center[0]), int(iris_center[1])

        cv2.circle(mask, (cx, cy), int(iris_radius * 1.05), 1.0, -1)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def _create_pupil_mask(
        self, w: int, h: int, iris_center: Tuple[float, float], iris_radius: float
    ) -> np.ndarray:
        """Create pupil (dark center) mask."""
        mask = np.zeros((h, w), dtype=np.float32)

        cx, cy = int(iris_center[0]), int(iris_center[1])
        pupil_r = iris_radius * 0.5

        cv2.circle(mask, (cx, cy), int(pupil_r), 1.0, -1)

        return mask

    def _create_eyelid_mask(
        self,
        lid_points: List[Tuple[float, float]],
        offset: Tuple[int, int],
        w: int,
        h: int,
        is_upper: bool,
    ) -> np.ndarray:
        """Create eyelid contour mask with smooth curve."""
        mask = np.zeros((h, w), dtype=np.float32)

        if len(lid_points) < 3:
            return mask

        points = np.array(
            [(p[0] - offset[0], p[1] - offset[1]) for p in lid_points], dtype=np.int32
        )

        points = points[np.argsort(points[:, 0])]

        if is_upper:
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 1.0)
        else:
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 1.0)

        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        return mask

    def _create_skin_mask(
        self, frame: np.ndarray, eye_bb: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Create skin mask around eye region."""
        x1, y1, x2, y2 = eye_bb
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return np.zeros((y2 - y1, x2 - x1), dtype=np.float32)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

        return skin_mask.astype(np.float32) / 255.0

    def _points_to_contour(
        self, points: List[Tuple[float, float]], offset: Tuple[int, int]
    ) -> np.ndarray:
        """Convert landmark points to contour array."""
        return np.array(
            [(int(p[0] - offset[0]), int(p[1] - offset[1])) for p in points],
            dtype=np.int32,
        )

    def _compute_segmentation_confidence(
        self,
        eye_region: np.ndarray,
        iris_mask: np.ndarray,
        upper_lid: np.ndarray,
        lower_lid: np.ndarray,
    ) -> float:
        """Compute confidence score for segmentation quality."""
        eye_area = np.sum(eye_region > 0) / (
            eye_region.shape[0] * eye_region.shape[1] + 1e-6
        )

        iris_area = np.sum(iris_mask > 0.5) / (
            iris_mask.shape[0] * iris_mask.shape[1] + 1e-6
        )

        lid_coverage = np.sum(upper_lid > 0.3) + np.sum(lower_lid > 0.3)
        lid_coverage = min(1.0, lid_coverage / (upper_lid.shape[0] * 2))

        confidence = 0.3 * eye_area + 0.4 * iris_area + 0.3 * lid_coverage

        return min(1.0, max(0.0, confidence))


class GazeEstimator:
    """
    Gaze estimation using iris position relative to eye corners.
    Provides realistic iris movement for Sharingan deformation.
    """

    def __init__(self):
        self._history = []
        self._max_history = 5

    def estimate_gaze(
        self, landmarks: List[Tuple[float, float]], eye_side: str = "left"
    ) -> Optional[GazeState]:
        """Estimate gaze direction from iris position."""
        if not landmarks or len(landmarks) < 478:
            return None

        if eye_side == "left":
            outer_corner = landmarks[33]
            inner_corner = landmarks[133]
        else:
            outer_corner = landmarks[362]
            inner_corner = landmarks[263]

        iris_center, iris_radius = self._get_iris_center_radius(landmarks, eye_side)

        eye_width = np.sqrt(
            (inner_corner[0] - outer_corner[0]) ** 2
            + (inner_corner[1] - outer_corner[1]) ** 2
        )

        eye_center_x = (outer_corner[0] + inner_corner[0]) / 2

        normalized_x = (iris_center[0] - eye_center_x) / (eye_width / 2 + 1e-6)
        normalized_y = (iris_center[1] - (outer_corner[1] + inner_corner[1]) / 2) / (
            eye_width / 2 + 1e-6
        )

        normalized_x = max(-1.5, min(1.5, normalized_x))
        normalized_y = max(-1.5, min(1.5, normalized_y))

        h_angle = np.degrees(np.arctan2(normalized_x, 1.0))
        v_angle = np.degrees(np.arctan2(-normalized_y, 1.0))

        direction = self._classify_direction(normalized_x, normalized_y)

        confidence = 1.0 - (abs(normalized_x) + abs(normalized_y)) / 3.0
        confidence = max(0.3, min(1.0, confidence))

        self._history.append((normalized_x, normalized_y))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        smoothed = np.mean(self._history, axis=0)

        return GazeState(
            direction=direction,
            horizontal_angle=h_angle,
            vertical_angle=v_angle,
            iris_offset=(float(smoothed[0]), float(smoothed[1])),
            confidence=confidence,
        )

    def _get_iris_center_radius(
        self, landmarks: List[Tuple[float, float]], eye_side: str
    ) -> Tuple[Tuple[float, float], float]:
        """Extract iris center and radius."""
        iris_idx = list(range(468, 478))

        iris_points = [landmarks[i] for i in iris_idx if i < len(landmarks)]

        cx = sum(p[0] for p in iris_points) / len(iris_points)
        cy = sum(p[1] for p in iris_points) / len(iris_points)

        max_dist = max(
            np.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in iris_points
        )

        return (cx, cy), max_dist

    def _classify_direction(self, norm_x: float, norm_y: float) -> str:
        """Classify gaze direction."""
        h_thresh = 0.25
        v_thresh = 0.25

        if abs(norm_x) < h_thresh and abs(norm_y) < v_thresh:
            return "center"
        elif norm_x < -h_thresh and abs(norm_y) < v_thresh:
            return "left"
        elif norm_x > h_thresh and abs(norm_y) < v_thresh:
            return "right"
        elif abs(norm_x) < h_thresh and norm_y < -v_thresh:
            return "up"
        elif abs(norm_x) < h_thresh and norm_y > v_thresh:
            return "down"
        elif norm_x < -h_thresh and norm_y < -v_thresh:
            return "up_left"
        elif norm_x > h_thresh and norm_y < -v_thresh:
            return "up_right"
        elif norm_x < -h_thresh and norm_y > v_thresh:
            return "down_left"
        elif norm_x > h_thresh and norm_y > v_thresh:
            return "down_right"

        return "center"


class EyeGeometryEstimator:
    """
    Estimates 3D eye geometry for perspective-correct compositing.
    """

    def __init__(self):
        self._history = []

    def estimate_geometry(
        self,
        landmarks: List[Tuple[float, float]],
        eye_side: str = "left",
        focal_length: float = 500.0,
    ) -> Optional[EyeGeometry]:
        """Estimate eye geometry from landmarks."""
        if not landmarks or len(landmarks) < 478:
            return None

        if eye_side == "left":
            outer = landmarks[33]
            inner = landmarks[133]
            upper = landmarks[159]
            lower = landmarks[145]
        else:
            outer = landmarks[362]
            inner = landmarks[263]
            upper = landmarks[386]
            lower = landmarks[373]

        eye_width = np.sqrt((inner[0] - outer[0]) ** 2 + (inner[1] - outer[1]) ** 2)
        eye_height = upper[1] - lower[1]

        eyeball_radius = eye_width * 0.45

        curvature = eye_height / (eye_width + 1e-6)

        tilt_rad = np.arctan2(inner[1] - outer[1], inner[0] - outer[0])
        tilt_deg = np.degrees(tilt_rad)

        compression_x = 1.0 - 0.15 * abs(np.sin(tilt_rad))
        compression_y = 1.0 - 0.1 * abs(np.cos(tilt_rad))

        depth_scale = focal_length / (focal_length + 50)

        return EyeGeometry(
            eyeball_radius=eyeball_radius,
            curvature_factor=curvature,
            compression_x=compression_x,
            compression_y=compression_y,
            tilt_angle=tilt_deg,
            depth_scale=depth_scale,
        )


class BlinkDetector:
    """
    Detects blinking and eye closure for texture deformation.
    """

    def __init__(self):
        self._history = []
        self._max_history = 10

    def detect_blink(
        self, landmarks: List[Tuple[float, float]], eye_side: str = "left"
    ) -> dict:
        """Detect blink state and openness."""
        if not landmarks or len(landmarks) < 478:
            return {"is_blinking": False, "openness": 1.0, "blink_speed": 0.0}

        if eye_side == "left":
            upper_idx = [33, 34, 35, 36, 37, 38, 156, 159]
            lower_idx = [41, 40, 39, 145, 144, 143, 153, 158]
        else:
            upper_idx = [263, 264, 265, 266, 267, 268, 384, 386]
            lower_idx = [271, 270, 269, 373, 374, 375, 381, 373]

        upper_y = min(landmarks[i][1] for i in upper_idx if i < len(landmarks))
        lower_y = max(landmarks[i][1] for i in lower_idx if i < len(landmarks))

        outer = landmarks[upper_idx[0] if eye_side == "left" else 263]
        inner = landmarks[upper_idx[-1] if eye_side == "left" else 263]
        eye_width = np.sqrt((inner[0] - outer[0]) ** 2 + (inner[1] - outer[1]) ** 2)

        eye_openness = (lower_y - upper_y) / (eye_width * 0.5 + 1e-6)
        eye_openness = max(0.0, min(1.0, eye_openness))

        self._history.append(eye_openness)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        blink_speed = 0.0
        if len(self._history) >= 3:
            speeds = [
                self._history[i] - self._history[i - 1]
                for i in range(1, len(self._history))
            ]
            blink_speed = np.mean(speeds)

        is_blinking = eye_openness < 0.2

        return {
            "is_blinking": is_blinking,
            "openness": eye_openness,
            "blink_speed": blink_speed,
            "is_closing": blink_speed < -0.05,
            "is_opening": blink_speed > 0.05,
        }
