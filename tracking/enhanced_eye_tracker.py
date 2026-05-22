"""
Professional VFX Eye Tracking Module
Provides full eye mesh, eyelid tracking, and 3D eye pose estimation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EyeMesh:
    """Complete eye mesh with eyelids and orientation."""

    iris_center: Tuple[float, float]
    iris_radius: float
    eye_center: Tuple[float, float]
    eye_width: float
    eye_height: float

    upper_eyelid: List[Tuple[float, float]]
    lower_eyelid: List[Tuple[float, float]]
    eye_corners: Tuple[Tuple[float, float], Tuple[float, float]]

    eye_angle: float
    eye_tilt: float

    openness: float
    blink_state: float

    confidence: float = 0.5


@dataclass
class EyePose3D:
    """3D eye pose for perspective-correct rendering."""

    rotation_vec: np.ndarray
    translation_vec: np.ndarray
    euler_angles: Tuple[float, float, float]
    confidence: float


class EnhancedEyeTracker:
    """
    Professional eye tracking for VFX applications.
    Extracts full eye mesh, eyelid contours, and 3D pose.
    """

    LEFT_EYE_INDICES = {
        "upper_lid": [33, 34, 35, 36, 37, 38],
        "lower_lid": [41, 40, 39, 42, 47, 46, 45, 44],
        "outer_corner": 33,
        "inner_corner": 133,
        "iris": 468,
    }

    RIGHT_EYE_INDICES = {
        "upper_lid": [263, 264, 265, 266, 267, 268],
        "lower_lid": [271, 270, 269, 272, 277, 276, 275, 274],
        "outer_corner": 362,
        "inner_corner": 263,
        "iris": 473,
    }

    IRIS_INDICES = list(range(468, 478))

    def __init__(self):
        self._smoothing_factor = 0.25
        self._pose_history = []
        self._max_history = 5
        self._last_iris_left = None
        self._last_iris_right = None
        self._iris_confidence_left = 0.0
        self._iris_confidence_right = 0.0

    def extract_eye_mesh(
        self, landmarks: List[Tuple[int, int]], side: str = "left"
    ) -> Optional[EyeMesh]:
        """Extract complete eye mesh from 468-point FaceMesh with robust iris detection."""
        if not landmarks or len(landmarks) < 478:
            return None

        side_data = self.LEFT_EYE_INDICES if side == "left" else self.RIGHT_EYE_INDICES

        upper_lid = [landmarks[i] for i in side_data["upper_lid"] if i < len(landmarks)]
        lower_lid = [landmarks[i] for i in side_data["lower_lid"] if i < len(landmarks)]

        if len(upper_lid) < 3 or len(lower_lid) < 3:
            return None

        outer_corner = landmarks[side_data["outer_corner"]]
        inner_corner = landmarks[side_data["inner_corner"]]

        iris_landmarks = [landmarks[i] for i in self.IRIS_INDICES if i < len(landmarks)]
        if len(iris_landmarks) < 5:
            return None

        iris_center_x = sum(p[0] for p in iris_landmarks) / len(iris_landmarks)
        iris_center_y = sum(p[1] for p in iris_landmarks) / len(iris_landmarks)

        distances = [
            np.sqrt((p[0] - iris_center_x) ** 2 + (p[1] - iris_center_y) ** 2)
            for p in iris_landmarks
        ]
        median_dist = np.median(distances)
        valid_points = [
            p for p, d in zip(iris_landmarks, distances) if d < median_dist * 2
        ]

        if len(valid_points) >= 3:
            iris_center_x = sum(p[0] for p in valid_points) / len(valid_points)
            iris_center_y = sum(p[1] for p in valid_points) / len(valid_points)

        iris_center = (float(iris_center_x), float(iris_center_y))

        if side == "left":
            if self._last_iris_left is not None:
                dist = np.sqrt(
                    (iris_center_x - self._last_iris_left[0]) ** 2
                    + (iris_center_y - self._last_iris_left[1]) ** 2
                )
                if dist > 20:
                    iris_center = self._last_iris_left
                    self._iris_confidence_left = max(
                        0.0, self._iris_confidence_left - 0.3
                    )
                else:
                    self._last_iris_left = iris_center
                    self._iris_confidence_left = min(
                        1.0, self._iris_confidence_left + 0.1
                    )
            else:
                self._last_iris_left = iris_center
                self._iris_confidence_left = 0.7
        else:
            if self._last_iris_right is not None:
                dist = np.sqrt(
                    (iris_center_x - self._last_iris_right[0]) ** 2
                    + (iris_center_y - self._last_iris_right[1]) ** 2
                )
                if dist > 20:
                    iris_center = self._last_iris_right
                    self._iris_confidence_right = max(
                        0.0, self._iris_confidence_right - 0.3
                    )
                else:
                    self._last_iris_right = iris_center
                    self._iris_confidence_right = min(
                        1.0, self._iris_confidence_right + 0.1
                    )
            else:
                self._last_iris_right = iris_center
                self._iris_confidence_right = 0.7

        max_dist = max(distances) if distances else 10
        iris_radius = float(max_dist)

        eye_width = np.sqrt(
            (outer_corner[0] - inner_corner[0]) ** 2
            + (outer_corner[1] - inner_corner[1]) ** 2
        )

        upper_y = min(p[1] for p in upper_lid)
        lower_y = max(p[1] for p in lower_lid)
        eye_height = lower_y - upper_y

        eye_center = ((outer_corner[0] + inner_corner[0]) / 2, (upper_y + lower_y) / 2)

        eye_angle = np.degrees(
            np.arctan2(
                inner_corner[1] - outer_corner[1], inner_corner[0] - outer_corner[0]
            )
        )

        eye_tilt = np.degrees(
            np.arctan2(
                (upper_lid[3][1] + upper_lid[2][1]) / 2
                - (lower_lid[3][1] + lower_lid[2][1]) / 2,
                outer_corner[0] - inner_corner[0],
            )
        )

        eye_height_expanded = eye_height * 1.2
        max_open = eye_height_expanded / eye_width if eye_width > 0 else 0.5
        openness = min(1.0, max_open / 0.4)

        blink_state = 1.0 - openness

        confidence = (
            self._iris_confidence_left
            if side == "left"
            else self._iris_confidence_right
        )

        return EyeMesh(
            iris_center=iris_center,
            iris_radius=iris_radius,
            eye_center=eye_center,
            eye_width=eye_width,
            eye_height=eye_height,
            upper_eyelid=upper_lid,
            lower_eyelid=lower_lid,
            eye_corners=(outer_corner, inner_corner),
            eye_angle=eye_angle,
            eye_tilt=eye_tilt,
            openness=openness,
            blink_state=blink_state,
            confidence=confidence,
        )

    def estimate_3d_pose(
        self, eye_mesh: EyeMesh, face_rotation: float = 0.0
    ) -> EyePose3D:
        """Estimate 3D eye pose for perspective-correct rendering."""

        rx = np.radians(face_rotation * 0.3)
        ry = np.radians(eye_mesh.eye_angle * 0.1)
        rz = np.radians(-eye_mesh.eye_tilt * 0.2)

        rotation_vec = np.array([rx, ry, rz])

        translation_vec = np.array([0, 0, 50])

        euler_angles = (np.degrees(rx), np.degrees(ry), np.degrees(rz))

        confidence = min(1.0, eye_mesh.openness + 0.2)

        self._pose_history.append((rotation_vec, translation_vec, confidence))
        if len(self._pose_history) > self._max_history:
            self._pose_history.pop(0)

        avg_rotation = np.mean([p[0] for p in self._pose_history], axis=0)
        avg_translation = np.mean([p[1] for p in self._pose_history], axis=0)
        avg_confidence = np.mean([p[2] for p in self._pose_history])

        return EyePose3D(
            rotation_vec=avg_rotation,
            translation_vec=avg_translation,
            euler_angles=euler_angles,
            confidence=avg_confidence,
        )

    def generate_eyelid_mask(
        self, eye_mesh: EyeMesh, frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Generate proper eyelid occlusion mask with soft edges."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.float32)

        upper_pts = np.array(
            self._smooth_contour(eye_mesh.upper_eyelid), dtype=np.int32
        )
        lower_pts = np.array(
            self._smooth_contour(eye_mesh.lower_eyelid), dtype=np.int32
        )

        if len(upper_pts) < 3 or len(lower_pts) < 3:
            return mask

        full_contour = np.vstack([upper_pts, lower_pts[::-1]])

        cv2.fillPoly(mask, [full_contour], 1.0)

        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    def _smooth_contour(
        self, points: List[Tuple[float, float]], iterations: int = 3
    ) -> List[Tuple[float, float]]:
        """Smooth contour points for cleaner edges."""
        if len(points) < 4:
            return points

        smoothed = points.copy()
        for _ in range(iterations):
            new_points = [smoothed[0]]
            for i in range(1, len(smoothed) - 1):
                x = (smoothed[i - 1][0] + 2 * smoothed[i][0] + smoothed[i + 1][0]) / 4
                y = (smoothed[i - 1][1] + 2 * smoothed[i][1] + smoothed[i + 1][1]) / 4
                new_points.append((x, y))
            new_points.append(smoothed[-1])
            smoothed = new_points

        return smoothed

    def get_iris_geometry(self, eye_mesh: EyeMesh) -> dict:
        """Get iris geometry for Sharingan placement."""
        cx, cy = eye_mesh.iris_center
        r = eye_mesh.iris_radius

        return {
            "center": (cx, cy),
            "radius": r,
            "visible_radius": r * eye_mesh.openness,
            "angle": np.radians(eye_mesh.eye_angle),
            "tilt": np.radians(eye_mesh.eye_tilt),
        }
