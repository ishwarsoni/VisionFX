"""Environmental immersion effects for edge darkening and tunnel vision."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from presentation.activation_sequence import SequenceState
from presentation.presentation_data import PresentationContext


class ImmersionLayer:
    """Adds subtle tunnel vision and peripheral pressure."""

    def __init__(self, immersion_strength: float):
        self.immersion_strength = immersion_strength
        self._cache_size: Tuple[int, int] = (0, 0)
        self._mask: Optional[np.ndarray] = None

    def apply(
        self, frame: np.ndarray, context: PresentationContext, sequence: SequenceState
    ) -> np.ndarray:
        intensity = max(
            sequence.eye_lock_intensity,
            sequence.buildup_intensity,
            sequence.corruption_intensity,
            sequence.transform_intensity,
        )
        intensity *= self.immersion_strength
        if intensity <= 0.001:
            return frame

        h, w = frame.shape[:2]
        self._ensure_mask(w, h)
        assert self._mask is not None

        blurred = cv2.GaussianBlur(frame, (0, 0), 6.5)
        mix = np.clip(self._mask * intensity, 0.0, 1.0)[..., None]
        return np.clip(
            frame.astype(np.float32) * (1.0 - mix) + blurred.astype(np.float32) * mix,
            0,
            255,
        ).astype(np.uint8)

    def _ensure_mask(self, width: int, height: int) -> None:
        if self._cache_size == (width, height):
            return
        self._cache_size = (width, height)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx * xx + yy * yy)
        mask = np.clip((radius - 0.15) / 0.85, 0.0, 1.0)
        self._mask = mask
