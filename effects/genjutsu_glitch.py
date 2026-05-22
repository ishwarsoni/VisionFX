"""Cinematic glitch and corruption effects."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from activation.activation_state import ActivationState
from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


class GenjutsuGlitchEffect(BaseEffect):
    """Gradually escalating frame corruption and chromatic aberration."""

    name = "genjutsu_glitch"
    priority = 30
    minimum_quality = 2

    def __init__(self, enabled: bool = True, intensity: float = 0.75):
        super().__init__(enabled=enabled)
        self.max_intensity = max(0.0, min(1.5, intensity))
        self.current_intensity = 0.0
        self.previous_frame: Optional[np.ndarray] = None
        self.pulse_boost = 0.0
        self.interrupt_boost = 0.0

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name == "on_activation_start":
            self.pulse_boost = 1.0
        elif event_name == "on_activation_complete":
            self.pulse_boost = 1.3
        elif event_name == "on_interrupt":
            self.interrupt_boost = 0.8
        elif event_name == "on_cooldown_end":
            self.pulse_boost = 0.0

    def update(self, context: EffectContext) -> None:
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        if context.activation_state in (
            ActivationState.ACTIVATING,
            ActivationState.POWER_ACTIVE,
        ):
            base = max(0.0, (progress - 0.18) / 0.82)
        else:
            base = 0.0

        if self.pulse_boost > 0.0:
            self.pulse_boost = max(0.0, self.pulse_boost - context.delta_ms / 260.0)
        if self.interrupt_boost > 0.0:
            self.interrupt_boost = max(
                0.0, self.interrupt_boost - context.delta_ms / 260.0
            )

        self.current_intensity = min(
            self.max_intensity,
            base * self.max_intensity
            + self.pulse_boost * 0.2
            + self.interrupt_boost * 0.25,
        )

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.enabled or self.current_intensity <= 0.02:
            self.previous_frame = frame.copy()
            return frame

        h, w = frame.shape[:2]
        intensity = self.current_intensity
        result = frame.copy()

        # RGB split
        shift_x = int(
            2 + intensity * 7.0 + 2.0 * math.sin(context.timestamp_ms / 120.0)
        )
        shift_y = int(intensity * 2.0)
        b, g, r = cv2.split(result)
        r = np.roll(r, shift_x, axis=1)
        b = np.roll(b, -shift_x, axis=1)
        g = np.roll(g, shift_y, axis=0)
        result = cv2.merge((b, g, r))

        # Frame tearing bands
        if intensity > 0.3:
            bands = 2 + int(intensity * 4)
            band_height = max(8, h // (bands * 4))
            for band_index in range(bands):
                top = int(
                    (band_index * 0.31 + (context.frame_index % 5) * 0.03) * h
                ) % max(1, h - band_height)
                offset = int(
                    math.sin(context.timestamp_ms / 70.0 + band_index)
                    * (4 + intensity * 14)
                )
                result[top : top + band_height] = np.roll(
                    result[top : top + band_height], offset, axis=1
                )

        # Temporal corruption / frame repetition
        if self.previous_frame is not None and intensity > 0.2:
            stripes = max(1, int(h * (0.04 + intensity * 0.08)))
            start = int(
                (math.sin(context.timestamp_ms / 240.0) * 0.5 + 0.5)
                * max(1, h - stripes)
            )
            result[start : start + stripes] = cv2.addWeighted(
                result[start : start + stripes],
                0.35,
                self.previous_frame[start : start + stripes],
                0.65,
                0.0,
            )

        # Scanlines
        scanline_strength = 0.12 + intensity * 0.18
        scanline = np.full((h, 1, 1), 1.0, dtype=np.float32)
        scanline[::2] = 1.0 - scanline_strength
        result = (result.astype(np.float32) * scanline).clip(0, 255).astype(np.uint8)

        self.previous_frame = frame.copy()
        return result
