"""Activation impact effect for awakening moments."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from activation.activation_state import ActivationState
from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


class ActivationImpactEffect(BaseEffect):
    """Short cinematic impact sequence triggered on activation start."""

    name = "activation_impact"
    priority = 10
    minimum_quality = 1

    def __init__(self, enabled: bool = True, strength: float = 0.9):
        super().__init__(enabled=enabled)
        self.strength = max(0.0, min(1.0, strength))
        self.active = False
        self.elapsed_ms = 0.0
        self.freeze_frame: Optional[np.ndarray] = None
        self.freeze_duration_ms = 120.0
        self.duration_ms = 420.0
        self.current_intensity = 0.0
        self.flash_boost = 0.0
        self.complete_boost = 0.0
        self.interrupt_boost = 0.0

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name == "on_activation_start":
            self.active = True
            self.elapsed_ms = 0.0
            self.freeze_frame = None
            self.flash_boost = 1.0
        elif event_name == "on_activation_complete":
            self.complete_boost = 1.0
        elif event_name == "on_interrupt":
            self.interrupt_boost = 1.0
            self.active = True
            self.elapsed_ms = 0.0
        elif event_name == "on_cooldown_end":
            self.active = False
            self.freeze_frame = None

    def update(self, context: EffectContext) -> None:
        if context.activation_state == ActivationState.ACTIVATING and not self.active:
            self.active = True

        if self.flash_boost > 0.0:
            self.flash_boost = max(0.0, self.flash_boost - context.delta_ms / 160.0)

        if self.complete_boost > 0.0:
            self.complete_boost = max(
                0.0, self.complete_boost - context.delta_ms / 260.0
            )

        if self.interrupt_boost > 0.0:
            self.interrupt_boost = max(
                0.0, self.interrupt_boost - context.delta_ms / 240.0
            )

        if self.active:
            self.elapsed_ms += context.delta_ms
            if (
                self.elapsed_ms >= self.duration_ms
                and context.activation_state
                not in (
                    ActivationState.ACTIVATING,
                    ActivationState.POWER_ACTIVE,
                )
            ):
                self.active = False
                self.freeze_frame = None

        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        self.current_intensity = max(
            self.strength * progress,
            self.flash_boost * 0.95,
            self.complete_boost * 0.85,
        )
        if self.interrupt_boost > 0.0:
            self.current_intensity = max(
                self.current_intensity, self.interrupt_boost * 0.7
            )

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.enabled or self.current_intensity <= 0.01:
            return frame

        if self.freeze_frame is None:
            self.freeze_frame = frame.copy()

        source = frame
        if self.elapsed_ms <= self.freeze_duration_ms and self.freeze_frame is not None:
            source = self.freeze_frame

        h, w = source.shape[:2]
        intensity = self.current_intensity

        zoom = 1.0 + 0.045 * intensity + 0.018 * math.sin(context.timestamp_ms / 85.0)
        new_w = max(1, int(w * zoom))
        new_h = max(1, int(h * zoom))
        resized = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x_offset = max(0, (new_w - w) // 2)
        y_offset = max(0, (new_h - h) // 2)
        cropped = resized[y_offset : y_offset + h, x_offset : x_offset + w]
        if cropped.shape[0] != h or cropped.shape[1] != w:
            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        shake_x = int(math.sin(context.timestamp_ms / 23.0) * intensity * 4.0)
        shake_y = int(math.cos(context.timestamp_ms / 31.0) * intensity * 3.0)
        transform = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        shaken = cv2.warpAffine(
            cropped, transform, (w, h), borderMode=cv2.BORDER_REFLECT101
        )

        flash = np.full_like(shaken, 255)
        flash_mix = min(0.42, 0.12 + intensity * 0.24)
        flashed = cv2.addWeighted(shaken, 1.0, flash, flash_mix, 0.0)

        contrast = 1.0 + intensity * 0.06
        brightness = int(10 * intensity)
        return cv2.convertScaleAbs(flashed, alpha=contrast, beta=brightness)
