"""Cinematic anime-inspired color grading."""

from __future__ import annotations

import cv2
import numpy as np

from activation.activation_state import ActivationState
from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


class ColorGradingEffect(BaseEffect):
    """Applies tonal shifts, bloom and activation desaturation."""

    name = "color_grading"
    priority = 40
    minimum_quality = 1

    def __init__(self, enabled: bool = True, bloom_strength: float = 0.12):
        super().__init__(enabled=enabled)
        self.bloom_strength = max(0.0, min(0.4, bloom_strength))
        self.current_intensity = 0.0

    def update(self, context: EffectContext) -> None:
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        self.current_intensity = progress

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.enabled or self.current_intensity <= 0.01:
            return frame

        intensity = self.current_intensity
        result = frame.astype(np.float32)

        lift = 3 + (1 - intensity) * 5
        result = result + lift

        contrast = 1.0 + intensity * 0.08
        mean = np.mean(result)
        result = mean + (result - mean) * contrast

        result = np.clip(result, 0, 255)

        result[:, :, 0] += 4 * intensity
        result[:, :, 1] += 1 * intensity

        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(
            np.float32
        )
        sat_scale = 1.0 - intensity * 0.08
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        bright = np.maximum(toned.astype(np.int16) - 220, 0).astype(np.uint8)
        if np.any(bright):
            bloom = cv2.GaussianBlur(bright, (0, 0), 9.0)
            toned = cv2.addWeighted(
                toned, 1.0, bloom, self.bloom_strength * intensity, 0.0
            )

        return toned
