"""Spatial warping and lens-like distortion effects."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from activation.activation_state import ActivationState
from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


class WorldDistortionEffect(BaseEffect):
    """Temporal wobble and spatial displacement for activation scenes."""

    name = "world_distortion"
    priority = 50
    minimum_quality = 2

    def __init__(self, enabled: bool = True, distortion_amount: float = 0.08):
        super().__init__(enabled=enabled)
        self.distortion_amount = max(0.0, min(0.25, distortion_amount))
        self.current_intensity = 0.0
        self._grid_cache_size: Tuple[int, int] = (0, 0)
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None

    def update(self, context: EffectContext) -> None:
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        if context.activation_state in (
            ActivationState.ACTIVATING,
            ActivationState.POWER_ACTIVE,
        ):
            self.current_intensity = progress * self.distortion_amount
        else:
            self.current_intensity = 0.0

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.enabled or self.current_intensity <= 0.005:
            return frame

        h, w = frame.shape[:2]
        self._ensure_grid(w, h)
        assert self._map_x is not None and self._map_y is not None

        phase = context.timestamp_ms / 140.0
        amplitude = self.current_intensity * min(w, h) * 0.03
        wobble_x = np.sin(self._map_y / 42.0 + phase) * amplitude
        wobble_y = np.cos(self._map_x / 58.0 + phase * 1.17) * amplitude * 0.42

        map_x = (self._map_x + wobble_x).astype(np.float32)
        map_y = (self._map_y + wobble_y).astype(np.float32)
        distorted = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        # Gentle radial lens distortion
        center_x = w * 0.5
        center_y = h * 0.5
        dx = (self._map_x - center_x) / max(1.0, center_x)
        dy = (self._map_y - center_y) / max(1.0, center_y)
        radius_sq = dx * dx + dy * dy
        lens = 1.0 + radius_sq * self.current_intensity * 0.015
        lens_map_x = (self._map_x + dx * lens * amplitude * 0.18).astype(np.float32)
        lens_map_y = (self._map_y + dy * lens * amplitude * 0.18).astype(np.float32)
        return cv2.remap(
            distorted,
            lens_map_x,
            lens_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _ensure_grid(self, width: int, height: int) -> None:
        if self._grid_cache_size == (width, height):
            return
        self._grid_cache_size = (width, height)
        x_coords = np.tile(np.arange(width, dtype=np.float32), (height, 1))
        y_coords = np.tile(
            np.arange(height, dtype=np.float32).reshape(height, 1), (1, width)
        )
        self._map_x = x_coords
        self._map_y = y_coords
