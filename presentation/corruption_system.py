"""Progressive system corruption overlays and signal interference."""

from __future__ import annotations

import math

import cv2
import numpy as np

from presentation.activation_sequence import SequenceState
from presentation.presentation_data import PresentationContext


class CorruptionSystem:
    """Applies gradual reality corruption and interference."""

    def __init__(self, corruption_intensity: float):
        self.corruption_intensity = corruption_intensity
        self.current_intensity = 0.0
        self.interrupt_boost = 0.0

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name == "on_activation_start":
            self.current_intensity = max(self.current_intensity, 0.12)
        elif event_name == "on_power_active":
            self.current_intensity = max(self.current_intensity, 0.8)
        elif event_name == "on_interrupt":
            self.interrupt_boost = 1.0
        elif event_name == "on_cooldown":
            self.current_intensity *= 0.75

    def update(self, context: PresentationContext, sequence: SequenceState) -> None:
        progress = max(sequence.corruption_intensity, sequence.transform_intensity)
        target = min(1.0, progress * self.corruption_intensity)
        if sequence.stage.value == "INTERRUPTED":
            target = max(target, self.interrupt_boost * 0.9)
        self.current_intensity = self._approach(
            self.current_intensity, target, context.delta_ms / 220.0
        )
        self.interrupt_boost = self._approach(
            self.interrupt_boost, 0.0, context.delta_ms / 140.0
        )

    def apply(
        self, frame: np.ndarray, context: PresentationContext, sequence: SequenceState
    ) -> np.ndarray:
        intensity = self.current_intensity
        if intensity <= 0.02:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        band_count = max(1, int(2 + intensity * 6))
        for band in range(band_count):
            band_y = int(((band * 0.19) + (context.frame_index % 5) * 0.07) * h) % max(
                1, h
            )
            band_h = max(2, int(h * (0.008 + intensity * 0.01)))
            shift = int(
                math.sin(context.timestamp_ms / 35.0 + band) * (4 + intensity * 12)
            )
            end_y = min(h, band_y + band_h)
            if end_y > band_y:
                result[band_y:end_y] = np.roll(result[band_y:end_y], shift, axis=1)

        split = int(2 + intensity * 6)
        b, g, r = cv2.split(result)
        r = np.roll(r, split, axis=1)
        b = np.roll(b, -split, axis=1)
        result = cv2.merge((b, g, r))

        flicker = np.full_like(result, 255)
        flicker_alpha = min(0.18, intensity * 0.14)
        if int(context.timestamp_ms / 70.0) % 4 == 0:
            result = cv2.addWeighted(result, 1.0, flicker, flicker_alpha, 0.0)

        return result

    @staticmethod
    def _approach(value: float, target: float, factor: float) -> float:
        factor = max(0.0, min(1.0, factor))
        return value + (target - value) * factor
