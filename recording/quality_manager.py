"""Adaptive quality scaling system with predefined quality profiles."""

from __future__ import annotations

from typing import Dict, List, Optional

from recording.recording_data import QualityMode, QualityProfile
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Predefined quality profiles
# ---------------------------------------------------------------------------

_QUALITY_PROFILES: Dict[QualityMode, QualityProfile] = {
    QualityMode.LOW: QualityProfile(
        mode=QualityMode.LOW,
        resolution_scale=0.5,
        compositor_quality=1,
        target_fps=30,
        effect_intensity_scale=0.5,
        label="LOW",
    ),
    QualityMode.MEDIUM: QualityProfile(
        mode=QualityMode.MEDIUM,
        resolution_scale=0.75,
        compositor_quality=1,
        target_fps=30,
        effect_intensity_scale=0.75,
        label="MEDIUM",
    ),
    QualityMode.HIGH: QualityProfile(
        mode=QualityMode.HIGH,
        resolution_scale=1.0,
        compositor_quality=2,
        target_fps=30,
        effect_intensity_scale=1.0,
        label="HIGH",
    ),
    QualityMode.CINEMATIC: QualityProfile(
        mode=QualityMode.CINEMATIC,
        resolution_scale=1.0,
        compositor_quality=3,
        target_fps=60,
        effect_intensity_scale=1.0,
        label="CINEMATIC",
    ),
}


class QualityManager:
    """
    Manages rendering quality tiers and adaptive quality recommendations.

    Does **not** mutate external configs directly — instead exposes a
    ``QualityProfile`` that the ``RecordingManager`` can read to adjust
    systems as needed.
    """

    def __init__(
        self,
        initial_mode: str = "HIGH",
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="QualityManager")
        self._mode = self._parse_mode(initial_mode)
        self._profile = _QUALITY_PROFILES[self._mode]
        self._fps_history: List[float] = []
        self._fps_window = 30

        self.logger.info(f"QualityManager initialized: {self._mode.value}")

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    @property
    def mode(self) -> QualityMode:
        return self._mode

    @property
    def profile(self) -> QualityProfile:
        return self._profile

    def set_mode(self, mode_str: str) -> QualityProfile:
        """Switch quality mode by name."""
        new_mode = self._parse_mode(mode_str)
        if new_mode != self._mode:
            self._mode = new_mode
            self._profile = _QUALITY_PROFILES[self._mode]
            self.logger.info(f"Quality mode changed to {self._mode.value}")
        return self._profile

    def get_profile(self, mode_str: Optional[str] = None) -> QualityProfile:
        """Retrieve a quality profile by mode name (or the current one)."""
        if mode_str is None:
            return self._profile
        target = self._parse_mode(mode_str)
        return _QUALITY_PROFILES[target]

    # ------------------------------------------------------------------
    # Adaptive quality
    # ------------------------------------------------------------------

    def auto_adjust(
        self, current_fps: float, target_fps: float = 30.0
    ) -> Optional[QualityProfile]:
        """Recommend a quality step-down if FPS is consistently below target."""
        self._fps_history.append(current_fps)
        if len(self._fps_history) > self._fps_window:
            self._fps_history = self._fps_history[-self._fps_window :]

        if len(self._fps_history) < self._fps_window:
            return None

        avg_fps = sum(self._fps_history) / len(self._fps_history)

        if avg_fps < target_fps * 0.85:
            lower = self._step_down(self._mode)
            if lower and lower != self._mode:
                self.logger.warning(
                    f"Auto-quality: FPS {avg_fps:.1f} < {target_fps * 0.85:.1f} "
                    f"-> recommending {lower.value}"
                )
                return _QUALITY_PROFILES[lower]

        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mode(value: str) -> QualityMode:
        try:
            return QualityMode(value.upper())
        except (ValueError, AttributeError):
            return QualityMode.HIGH

    @staticmethod
    def _step_down(mode: QualityMode) -> Optional[QualityMode]:
        order = [
            QualityMode.LOW,
            QualityMode.MEDIUM,
            QualityMode.HIGH,
            QualityMode.CINEMATIC,
        ]
        idx = order.index(mode) if mode in order else 2
        if idx > 0:
            return order[idx - 1]
        return None

    def get_debug_lines(self) -> List[str]:
        avg = 0.0
        if self._fps_history:
            avg = sum(self._fps_history) / len(self._fps_history)
        return [
            f"QUALITY: {self._mode.value}",
            f"RES SCALE: {self._profile.resolution_scale:.2f}",
            f"COMPOSITOR: Q{self._profile.compositor_quality}",
            f"AVG FPS: {avg:.1f}",
        ]
