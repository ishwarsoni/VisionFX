"""Base classes and helpers for composable visual effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from effects.effect_data import EffectContext


class BaseEffect(ABC):
    """Base class for real-time frame effects."""

    name: str = "base"
    priority: int = 100
    minimum_quality: int = 1

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def update(self, context: EffectContext) -> None:
        """Update internal effect state before rendering."""

    def handle_event(self, event_name: str, payload: dict) -> None:
        """Receive activation events routed by the effect manager."""

    @abstractmethod
    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        """Apply the effect to a frame."""

    def is_active(self, context: EffectContext) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @staticmethod
    def alpha_blend(
        base_frame: np.ndarray, overlay_frame: np.ndarray, alpha: float
    ) -> np.ndarray:
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 0.0:
            return base_frame
        if alpha >= 1.0:
            return overlay_frame
        return cv2.addWeighted(base_frame, 1.0 - alpha, overlay_frame, alpha, 0.0)

    @staticmethod
    def ensure_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)
