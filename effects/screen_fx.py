"""Lightweight finishing layer for vignette, grain, and pulse accents."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from activation.activation_state import ActivationState
from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


class ScreenFxEffect(BaseEffect):
    """Final layer with vignette, grain, and pulse flashes."""

    name = "screen_fx"
    priority = 60
    minimum_quality = 1

    def __init__(self, enabled: bool = True, intensity: float = 0.65):
        super().__init__(enabled=enabled)
        self.intensity = max(0.0, min(1.0, intensity))
        self.current_intensity = 0.0
        self.flash_boost = 0.0
        self._vignette_cache_size: Tuple[int, int] = (0, 0)
        self._vignette_mask: Optional[np.ndarray] = None

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name == "on_activation_start":
            self.flash_boost = 1.0
        elif event_name == "on_activation_complete":
            self.flash_boost = 1.35
        elif event_name == "on_interrupt":
            self.flash_boost = 0.65

    def update(self, context: EffectContext) -> None:
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        if context.activation_state in (
            ActivationState.ACTIVATING,
            ActivationState.POWER_ACTIVE,
        ):
            self.current_intensity = progress * self.intensity
        else:
            self.current_intensity = progress * self.intensity * 0.35

        if self.flash_boost > 0.0:
            self.flash_boost = max(0.0, self.flash_boost - context.delta_ms / 160.0)

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.enabled or self.current_intensity <= 0.01:
            return frame

        h, w = frame.shape[:2]
        self._ensure_vignette(w, h)
        assert self._vignette_mask is not None

        result = frame.copy()

        # Vignette / edge darkening
        darkened = (result.astype(np.float32) * self._vignette_mask[..., None]).astype(
            np.uint8
        )
        result = darkened

        # Film grain, low-cost deterministic noise
        grain_strength = int(5 + self.current_intensity * 18)
        if grain_strength > 0:
            rng = np.random.default_rng(
                int(context.frame_index * 13 + context.timestamp_ms // 17)
            )
            grain_small = rng.integers(
                -grain_strength,
                grain_strength + 1,
                size=(max(1, h // 3), max(1, w // 3)),
                dtype=np.int16,
            )
            grain = cv2.resize(
                grain_small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
            )
            grain = np.repeat(grain[:, :, None], 3, axis=2)
            result = np.clip(
                result.astype(np.int16) + grain.astype(np.int16), 0, 255
            ).astype(np.uint8)

        # Pulse flash accents
        if self.flash_boost > 0.0:
            flash_overlay = np.full_like(result, 255)
            flash_alpha = min(0.28, 0.08 + self.flash_boost * 0.14)
            result = cv2.addWeighted(result, 1.0, flash_overlay, flash_alpha, 0.0)

        return result

    def _ensure_vignette(self, width: int, height: int) -> None:
        if self._vignette_cache_size == (width, height):
            return
        self._vignette_cache_size = (width, height)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx * xx + yy * yy)
        mask = np.clip(1.0 - radius * 0.72, 0.58, 1.0)
        self._vignette_mask = mask
