"""Cinematic camera motion and subtle focus simulation."""

from __future__ import annotations

import math

import cv2
import numpy as np

from presentation.activation_sequence import SequenceState
from presentation.presentation_data import PresentationContext


class CinematicCamera:
    """Applies gentle zoom, drift, shake, and focus pulses."""

    def __init__(self, zoom_strength: float):
        self.zoom_strength = zoom_strength

    def apply(
        self, frame: np.ndarray, context: PresentationContext, sequence: SequenceState
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        intensity = max(
            sequence.eye_lock_intensity,
            sequence.buildup_intensity,
            sequence.transform_intensity,
            sequence.interruption_intensity,
        )
        if intensity <= 0.001:
            return frame

        zoom = (
            1.0
            + self.zoom_strength * intensity
            + 0.01 * math.sin(context.timestamp_ms / 420.0)
        )
        zoom = max(1.0, zoom)
        new_w = max(1, int(w * zoom))
        new_h = max(1, int(h * zoom))
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        drift_x = int(math.sin(context.timestamp_ms / 720.0) * w * 0.01 * intensity)
        drift_y = int(math.cos(context.timestamp_ms / 980.0) * h * 0.01 * intensity)
        x_offset = max(0, (new_w - w) // 2 + drift_x)
        y_offset = max(0, (new_h - h) // 2 + drift_y)
        crop = scaled[y_offset : y_offset + h, x_offset : x_offset + w]
        if crop.shape[:2] != (h, w):
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        shake_x = int(math.sin(context.timestamp_ms / 19.0) * 2.0 * intensity)
        shake_y = int(math.cos(context.timestamp_ms / 27.0) * 1.6 * intensity)
        matrix = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        shaken = cv2.warpAffine(crop, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)

        blur_amount = 1.0 + intensity * 1.2
        blurred = cv2.GaussianBlur(shaken, (0, 0), blur_amount)
        focus_mix = min(0.28, intensity * 0.18)
        return cv2.addWeighted(shaken, 1.0 - focus_mix, blurred, focus_mix, 0.0)
