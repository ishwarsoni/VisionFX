"""
Debug visualization for tracking data.
Renders tracking information on frames for visualization.
Maintains separation between detection logic and rendering.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from tracking.tracking_data import TrackingFrameData, TrackingState
from utils.logger import Logger


class TrackingDebugRenderer:
    """
    Professional debug visualization for tracking data.
    Renders iris markers, face boxes, gaze directions, and statistics.
    """

    # Visual configuration
    FACE_BOX_COLOR = (0, 255, 200)  # Cyan
    IRIS_COLOR = (255, 180, 80)  # Blue-tinted highlight
    IRIS_RADIUS = 8
    IRIS_OUTLINE_THICKNESS = 2

    LANDMARK_COLOR = (255, 220, 120)  # Blue-tinted highlight
    LANDMARK_RADIUS = 2

    GAZE_INDICATOR_LENGTH = 40
    GAZE_COLOR = (0, 255, 0)  # Green

    INFO_PANEL_BG = (20, 20, 40)
    INFO_PANEL_COLOR = (0, 255, 200)  # Cyan

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (0, 255, 200)

    def __init__(self, enabled: bool = True, logger: Optional[Logger] = None):
        """
        Initialize tracking debug renderer.

        Args:
            enabled: Whether rendering is initially enabled
            logger: Logger instance
        """
        self.enabled = enabled
        self.logger = logger or Logger(name="TrackingDebugRenderer")
        self.logger.info("TrackingDebugRenderer initialized")

    def render(self, frame: np.ndarray, tracking_data: TrackingFrameData) -> np.ndarray:
        """
        Render debug visualization on frame.

        Args:
            frame: Input frame
            tracking_data: Tracking data to visualize

        Returns:
            Frame with overlays rendered
        """
        if not self.enabled:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        # Draw face box if detected
        if tracking_data.face.detected:
            self._draw_face_box(result, tracking_data.face)
            self._draw_landmarks(result, tracking_data.face.landmarks)

        # Draw iris markers if tracking
        if tracking_data.state != TrackingState.NO_FACE:
            self._draw_iris_markers(result, tracking_data, w, h)
            self._draw_gaze_indicators(result, tracking_data, w, h)

        # Draw debug panel
        self._draw_info_panel(result, tracking_data)

        return result

    def _draw_face_box(self, frame: np.ndarray, face_data) -> None:
        """Draw face bounding box."""
        x, y, w, h = face_data.bounding_box

        if w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.FACE_BOX_COLOR, 2)

            # Draw confidence text
            conf_text = f"Face: {face_data.confidence:.2f}"
            cv2.putText(
                frame,
                conf_text,
                (x, y - 10),
                self.FONT,
                self.FONT_SCALE,
                self.FACE_BOX_COLOR,
                self.FONT_THICKNESS,
            )

    def _draw_landmarks(
        self, frame: np.ndarray, landmarks: list, stride: int = 3
    ) -> None:
        """Draw face landmarks."""
        for i, (x, y) in enumerate(landmarks):
            if i % stride == 0:  # Draw every nth landmark to avoid clutter
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    self.LANDMARK_RADIUS,
                    self.LANDMARK_COLOR,
                    -1,
                )

    def _draw_iris_markers(
        self,
        frame: np.ndarray,
        tracking_data: TrackingFrameData,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Draw iris center markers."""
        left_iris_x, left_iris_y = self._iris_to_frame_point(
            tracking_data.left_eye.landmarks,
            tracking_data.left_eye.iris.center,
            frame_w,
            frame_h,
            fallback_center=tracking_data.face.center,
        )

        right_iris_x, right_iris_y = self._iris_to_frame_point(
            tracking_data.right_eye.landmarks,
            tracking_data.right_eye.iris.center,
            frame_w,
            frame_h,
            fallback_center=tracking_data.face.center,
        )

        # Clamp to frame bounds
        left_iris_x = max(0, min(frame_w - 1, left_iris_x))
        left_iris_y = max(0, min(frame_h - 1, left_iris_y))
        right_iris_x = max(0, min(frame_w - 1, right_iris_x))
        right_iris_y = max(0, min(frame_h - 1, right_iris_y))

        # Draw circles
        cv2.circle(
            frame,
            (left_iris_x, left_iris_y),
            self.IRIS_RADIUS,
            self.IRIS_COLOR,
            self.IRIS_OUTLINE_THICKNESS,
        )
        cv2.circle(
            frame,
            (right_iris_x, right_iris_y),
            self.IRIS_RADIUS,
            self.IRIS_COLOR,
            self.IRIS_OUTLINE_THICKNESS,
        )

        # Draw crosshair for eye contact visualization
        if tracking_data.eye_contact.in_contact:
            cv2.line(
                frame,
                (left_iris_x - 5, left_iris_y),
                (left_iris_x + 5, left_iris_y),
                (0, 255, 0),
                1,
            )
            cv2.line(
                frame,
                (left_iris_x, left_iris_y - 5),
                (left_iris_x, left_iris_y + 5),
                (0, 255, 0),
                1,
            )
            cv2.line(
                frame,
                (right_iris_x - 5, right_iris_y),
                (right_iris_x + 5, right_iris_y),
                (0, 255, 0),
                1,
            )
            cv2.line(
                frame,
                (right_iris_x, right_iris_y - 5),
                (right_iris_x, right_iris_y + 5),
                (0, 255, 0),
                1,
            )

    def _iris_to_frame_point(
        self,
        eye_landmarks: list,
        iris_center: Tuple[float, float],
        frame_w: int,
        frame_h: int,
        fallback_center: Tuple[float, float],
    ) -> Tuple[int, int]:
        """Map normalized iris coordinates to the frame using the eye landmark ROI."""
        if eye_landmarks:
            xs = [x for x, _ in eye_landmarks]
            ys = [y for _, y in eye_landmarks]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            iris_x = int(x_min + iris_center[0] * max(1, x_max - x_min))
            iris_y = int(y_min + iris_center[1] * max(1, y_max - y_min))
        else:
            iris_x = int(fallback_center[0])
            iris_y = int(fallback_center[1])

        iris_x = max(0, min(frame_w - 1, iris_x))
        iris_y = max(0, min(frame_h - 1, iris_y))
        return iris_x, iris_y

    def _draw_gaze_indicators(
        self,
        frame: np.ndarray,
        tracking_data: TrackingFrameData,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Draw gaze direction indicators."""
        # Get iris centers
        face_x, face_y = int(tracking_data.face.center[0]), int(
            tracking_data.face.center[1]
        )

        # Left gaze direction
        left_gaze = tracking_data.left_eye.iris.gaze_direction
        if left_gaze.value != "CENTER":
            gaze_text = left_gaze.value
            cv2.putText(
                frame,
                f"L:{gaze_text}",
                (face_x - 60, face_y - 30),
                self.FONT,
                0.4,
                self.GAZE_COLOR,
                1,
            )

        # Right gaze direction
        right_gaze = tracking_data.right_eye.iris.gaze_direction
        if right_gaze.value != "CENTER":
            gaze_text = right_gaze.value
            cv2.putText(
                frame,
                f"R:{gaze_text}",
                (face_x + 30, face_y - 30),
                self.FONT,
                0.4,
                self.GAZE_COLOR,
                1,
            )

    def _draw_info_panel(
        self, frame: np.ndarray, tracking_data: TrackingFrameData
    ) -> None:
        """Draw information panel with tracking statistics."""
        h, w = frame.shape[:2]

        # Build info lines
        lines = []
        lines.append(f"State: {tracking_data.state.value}")
        lines.append(f"Conf: {tracking_data.overall_confidence:.2f}")

        # Eye status
        left_open = tracking_data.left_eye.openness
        right_open = tracking_data.right_eye.openness
        lines.append(f"Eyes: L:{left_open:.2f} R:{right_open:.2f}")

        # Blink status
        if tracking_data.blink.is_blinking:
            lines.append(f"● BLINKING ({tracking_data.blink.blink_duration_ms:.0f}ms)")
        else:
            lines.append(f"Blinks: {tracking_data.blink.blink_count}")

        # Eye contact
        if tracking_data.eye_contact.in_contact:
            lines.append(
                f"● EYE CONTACT ({tracking_data.eye_contact.duration_ms:.0f}ms)"
            )

        # Gaze direction
        left_gaze = tracking_data.left_eye.iris.gaze_direction.value
        right_gaze = tracking_data.right_eye.iris.gaze_direction.value
        if left_gaze == right_gaze:
            lines.append(f"Gaze: {left_gaze}")
        else:
            lines.append(f"Gaze: L:{left_gaze} R:{right_gaze}")

        # Draw panel background
        panel_height = len(lines) * 22 + 20
        overlay = frame.copy()
        y_offset = max(10, h - panel_height - 10)
        cv2.rectangle(
            overlay,
            (10, y_offset),
            (300, y_offset + panel_height),
            self.INFO_PANEL_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (10, y_offset),
            (300, y_offset + panel_height),
            self.INFO_PANEL_COLOR,
            2,
        )

        # Draw text
        for idx, line in enumerate(lines):
            y_pos = y_offset + 20 + (idx * 22)
            cv2.putText(
                frame,
                line,
                (20, y_pos),
                self.FONT,
                self.FONT_SCALE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )

    def toggle(self) -> None:
        """Toggle rendering on/off."""
        self.enabled = not self.enabled
        self.logger.info(f"Debug rendering {'enabled' if self.enabled else 'disabled'}")

    def set_enabled(self, enabled: bool) -> None:
        """Set rendering enabled state."""
        self.enabled = enabled
