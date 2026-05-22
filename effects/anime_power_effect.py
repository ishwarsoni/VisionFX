"""
Anime Power Effect Manager - Clean implementation
"""

from enum import Enum

import cv2
import numpy as np

from effects.base_effect import BaseEffect
from effects.chidori_effect import ChidoriEffect, create_chidori
from effects.effect_data import EffectContext
from effects.rasengan_effect import RasenganEffect, create_rasengan


class PowerState(Enum):
    NO_HAND = "no_hand"
    HAND_DETECTED = "hand_detected"
    RASENGAN_ACTIVE = "rasengan_active"
    CHIDORI_ACTIVE = "chidori_active"
    FUSION_ACTIVE = "fusion_active"


class AnimePowerEffect(BaseEffect):
    """Clean hand gesture detection and power management."""

    name = "anime_power"
    priority = 20

    # Hand connections for skeleton drawing
    CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

        self.rasengan = create_rasengan()
        self.chidori = create_chidori()

        self._hand_detector = None
        self._state = PowerState.NO_HAND
        self._current_power = None
        self._left_hand_data = None
        self._right_hand_data = None
        self._left_landmarks = None
        self._right_landmarks = None

    def set_hand_detector(self, detector):
        self._hand_detector = detector

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        """Process frame with hand detection and power effects."""
        h, w = frame.shape[:2]

        if not self.enabled:
            return frame

        if not self._hand_detector:
            return frame

        raw_frame = context.raw_frame if context.raw_frame is not None else frame
        hand_results = (
            self._hand_detector.process_hands(raw_frame)
            if hasattr(self._hand_detector, "process_hands")
            else None
        )
        if hand_results is None:
            single_landmarks = self._hand_detector.process_frame(raw_frame)
            hand_results = (
                [
                    {
                        "landmarks": single_landmarks,
                        "handedness": "unknown",
                        "handedness_confidence": 0.0,
                    }
                ]
                if single_landmarks
                else []
            )

        if context.debug_enabled:
            debug_info = self._hand_detector.get_debug_info()
            cv2.putText(
                frame,
                debug_info,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if not hand_results:
            self._clear_hand_state()
            if context.debug_enabled:
                cv2.putText(
                    frame,
                    "NO HAND",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            return frame

        hands = (
            self._hand_detector.detect_hands(hand_results, (h, w))
            if hasattr(self._hand_detector, "detect_hands")
            else []
        )
        if not hands:
            for item in hand_results:
                lms = item.get("landmarks") if isinstance(item, dict) else None
                if lms is None:
                    continue
                hd = self._hand_detector.detect_hand(lms, (h, w))
                if hd:
                    hd.handedness = str(item.get("handedness", "unknown")).lower()
                    hands.append(hd)

        left_candidates = []
        right_candidates = []
        for idx, hand in enumerate(hands):
            if not self._is_valid_hand(hand, context):
                continue
            label = (hand.handedness or "unknown").lower()
            if label == "left":
                left_candidates.append((hand.confidence, idx, hand))
            elif label == "right":
                right_candidates.append((hand.confidence, idx, hand))

        self._left_hand_data = max(left_candidates, default=(0.0, -1, None))[2]
        self._right_hand_data = max(right_candidates, default=(0.0, -1, None))[2]

        left_idx = max(left_candidates, default=(0.0, -1, None))[1]
        right_idx = max(right_candidates, default=(0.0, -1, None))[1]
        self._left_landmarks = (
            hand_results[left_idx]["landmarks"] if left_idx >= 0 else None
        )
        self._right_landmarks = (
            hand_results[right_idx]["landmarks"] if right_idx >= 0 else None
        )

        if self._left_hand_data and self._right_hand_data:
            self._activate_rasengan(self._left_hand_data)
            self._activate_chidori(self._right_hand_data)
            self._state = PowerState.FUSION_ACTIVE
            self._current_power = "fusion"
        elif self._left_hand_data:
            self._activate_rasengan(self._left_hand_data)
            self.chidori.deactivate()
            self._state = PowerState.RASENGAN_ACTIVE
            self._current_power = "rasengan"
        elif self._right_hand_data:
            self._activate_chidori(self._right_hand_data)
            self.rasengan.deactivate()
            self._state = PowerState.CHIDORI_ACTIVE
            self._current_power = "chidori"
        else:
            self._clear_hand_state()
            return frame

        if context.debug_enabled:
            if self._left_landmarks is not None:
                self._draw_hand_skeleton(frame, self._left_landmarks, h, w)
            if self._right_landmarks is not None:
                self._draw_hand_skeleton(frame, self._right_landmarks, h, w)
            self._draw_debug_overlay(frame)

        if self._left_hand_data:
            dt = context.delta_ms / 1000.0
            vel = self._left_hand_data.velocity
            self.rasengan.update(self._left_hand_data.palm_center, dt, vel)
            frame = self.rasengan.render(frame, self._left_hand_data.palm_center)

        if self._right_hand_data and self._right_landmarks is not None:
            dt = context.delta_ms / 1000.0
            vel = self._right_hand_data.velocity
            self.chidori.update(
                self._right_hand_data.palm_center,
                dt,
                vel,
                self._right_landmarks,
                (h, w),
            )
            frame = self.chidori.render(
                frame, self._right_hand_data.palm_center, self._right_landmarks
            )

        if context.debug_enabled:
            cv2.putText(
                frame,
                f"State: {self._state.value}",
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        return frame

    def _draw_hand_skeleton(self, frame, landmarks, h, w):
        """Draw hand skeleton and landmarks."""
        if not landmarks:
            return

        # Draw landmarks
        if hasattr(landmarks, "landmark"):
            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Draw connections
        for start, end in self.CONNECTIONS:
            if hasattr(landmarks, "landmark"):
                if start < len(landmarks.landmark) and end < len(landmarks.landmark):
                    x1 = int(landmarks.landmark[start].x * w)
                    y1 = int(landmarks.landmark[start].y * h)
                    x2 = int(landmarks.landmark[end].x * w)
                    y2 = int(landmarks.landmark[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    def _draw_debug_overlay(self, frame) -> None:
        """Draw handedness-based power assignment status."""
        left_state = "ON" if self._left_hand_data else "OFF"
        right_state = "ON" if self._right_hand_data else "OFF"
        mode = self._current_power or "none"

        cv2.putText(
            frame,
            f"LEFT HAND -> RASENGAN: {left_state}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 210, 120),
            2,
        )
        cv2.putText(
            frame,
            f"RIGHT HAND -> CHIDORI: {right_state}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 210, 120),
            2,
        )
        cv2.putText(
            frame,
            f"MODE: {mode}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 240, 255),
            2,
        )

        if self._left_hand_data:
            lx, ly = int(self._left_hand_data.palm_center[0]), int(
                self._left_hand_data.palm_center[1]
            )
            cv2.circle(frame, (lx, ly), 8, (255, 190, 90), -1)
            cv2.putText(
                frame,
                "L",
                (lx - 5, ly - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 160),
                2,
            )
        if self._right_hand_data:
            rx, ry = int(self._right_hand_data.palm_center[0]), int(
                self._right_hand_data.palm_center[1]
            )
            cv2.circle(frame, (rx, ry), 8, (255, 245, 220), -1)
            cv2.putText(
                frame,
                "R",
                (rx - 5, ry - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    def _set_hand_detected(self) -> None:
        if self._state in (
            PowerState.RASENGAN_ACTIVE,
            PowerState.CHIDORI_ACTIVE,
            PowerState.FUSION_ACTIVE,
        ):
            return
        self._state = PowerState.HAND_DETECTED
        self._current_power = None

    def _update_gesture_state(
        self,
        gesture_name: str,
        gesture_confidence: float,
        hand_data,
        context: EffectContext,
    ) -> None:
        # Intentionally unused: power assignment is based only on handedness.
        return

    def _activate_rasengan(self, hand_data) -> None:
        if not self.rasengan._active:
            self.rasengan.activate()

    def _activate_chidori(self, hand_data) -> None:
        if not self.chidori._active:
            self.chidori.activate()

    def _deactivate_power(self) -> None:
        self.rasengan.deactivate()
        self.chidori.deactivate()
        self._state = PowerState.NO_HAND
        self._current_power = None
        self._left_hand_data = None
        self._right_hand_data = None
        self._left_landmarks = None
        self._right_landmarks = None

    def _clear_hand_state(self) -> None:
        self._deactivate_power()

    def _is_valid_hand(self, hand_data, context: EffectContext) -> bool:
        if hand_data.confidence < 0.5:
            return False

        palm_x, palm_y = hand_data.palm_center
        if palm_x <= 0 or palm_y <= 0:
            return False

        return True

    def _classify_gesture(
        self, landmarks, hand_data, frame_shape
    ) -> tuple[str | None, float]:
        # Removed by design: handedness now directly controls powers.
        return None, 0.0

    def _extract_points(self, landmarks, frame_shape):
        h, w = frame_shape
        if hasattr(landmarks, "landmark"):
            return [
                np.array([lm.x * w, lm.y * h], dtype=np.float32)
                for lm in landmarks.landmark
            ]
        points = []
        for lm in landmarks:
            if hasattr(lm, "x"):
                points.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
                continue
            arr = np.array(lm, dtype=np.float32)
            if arr.shape[0] >= 2 and float(np.max(np.abs(arr[:2]))) <= 2.0:
                points.append(np.array([arr[0] * w, arr[1] * h], dtype=np.float32))
            else:
                points.append(arr[:2])
        return points

    def _angle(self, a, b, c) -> float:
        ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom <= 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _finger_extended(self, mcp, pip, tip, wrist) -> bool:
        angle = self._angle(mcp, pip, tip)
        tip_lift = float(pip[1] - tip[1])
        palm_lift = float(wrist[1] - tip[1])
        return angle >= 145.0 and tip_lift > 2.0 and palm_lift > 10.0

    def _thumb_extended(self, points) -> bool:
        angle = self._angle(points[1], points[2], points[4])
        thumb_span = float(
            np.linalg.norm(np.asarray(points[4]) - np.asarray(points[2]))
        )
        return angle >= 135.0 and thumb_span > 18.0

    def _finger_score(self, mcp, pip, dip, tip, palm_center) -> float:
        pip_angle = self._angle(mcp, pip, dip)
        dip_angle = self._angle(pip, dip, tip)
        reach = float(np.linalg.norm(np.asarray(tip) - np.asarray(palm_center)))
        palm_span = max(
            1e-6, float(np.linalg.norm(np.asarray(mcp) - np.asarray(palm_center)))
        )

        straightness = max(0.0, min(1.0, (pip_angle - 100.0) / 80.0))
        distal = max(0.0, min(1.0, (dip_angle - 100.0) / 80.0))
        extension = max(0.0, min(1.0, (reach / palm_span - 0.55) / 0.8))
        return float(0.45 * straightness + 0.25 * distal + 0.30 * extension)

    def _gesture_scores(self, points, palm_center):
        thumb = self._finger_score(
            points[1], points[2], points[3], points[4], palm_center
        )
        index = self._finger_score(
            points[5], points[6], points[7], points[8], palm_center
        )
        middle = self._finger_score(
            points[9], points[10], points[11], points[12], palm_center
        )
        ring = self._finger_score(
            points[13], points[14], points[15], points[16], palm_center
        )
        pinky = self._finger_score(
            points[17], points[18], points[19], points[20], palm_center
        )
        return thumb, index, middle, ring, pinky

    def _rasengan_confidence(self, states, points) -> float:
        thumb, index, middle, ring, pinky = states
        extended = sum((index, middle, ring, pinky))
        spread = float(
            np.linalg.norm(np.asarray(points[8]) - np.asarray(points[12]))
            + np.linalg.norm(np.asarray(points[12]) - np.asarray(points[16]))
            + np.linalg.norm(np.asarray(points[16]) - np.asarray(points[20]))
        )
        spread_score = max(0.0, min(1.0, (spread - 70.0) / 90.0))
        return float(
            0.5 * (extended / 4.0) + 0.3 * spread_score + 0.2 * (0.2 if thumb else 1.0)
        )

    def _chidori_confidence(self, states, points) -> float:
        thumb, index, middle, ring, pinky = states
        index_line = float(
            np.linalg.norm(np.asarray(points[8]) - np.asarray(points[6]))
        )
        folded_count = sum((not middle, not ring, not pinky))
        return float(
            0.55 * (1.0 if index else 0.0)
            + 0.25 * (folded_count / 3.0)
            + 0.20 * max(0.0, min(1.0, (index_line - 25.0) / 80.0))
        )


def create_anime_power_effect() -> AnimePowerEffect:
    return AnimePowerEffect()
