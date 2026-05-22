"""
Cinematic Activation Sequence System
Professional VFX transformation sequence with freeze-frame, glitch, and ignition effects.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import cv2
import numpy as np


class ActivationPhase(Enum):
    """Phases of the cinematic transformation sequence."""

    IDLE = "idle"
    STARE_DETECTED = "stare_detected"
    TENSION_BUILD = "tension_build"
    FREEZE_IMPACT = "freeze_impact"
    GLITCH_PULSE = "glitch_pulse"
    EYE_IGNITION = "eye_ignition"
    FULL_TRANSFORM = "full_transform"
    SUSTAIN = "sustain"
    COOLDOWN = "cooldown"


@dataclass
class CinematicSequenceConfig:
    """Configuration for cinematic activation sequence."""

    tension_duration_ms: float = 800.0
    freeze_duration_ms: float = 50.0
    glitch_duration_ms: float = 100.0
    ignition_duration_ms: float = 150.0
    transform_ramp_ms: float = 400.0

    enable_freeze_frame: bool = True
    enable_glitch_pulse: bool = True
    enable_ignition_spark: bool = True

    freeze_color_shift: tuple = (180, 30, 30)
    glitch_intensity: float = 0.4
    spark_count: int = 5

    tension_min_intensity: float = 0.1
    tension_max_intensity: float = 0.35


class CinematicActivationSystem:
    """
    Professional cinematic activation sequence.
    Creates the dramatic transformation moment with proper timing.
    """

    def __init__(self, config: Optional[CinematicSequenceConfig] = None):
        self.config = config or CinematicSequenceConfig()

        self.current_phase = ActivationPhase.IDLE
        self.phase_time_ms = 0.0
        self.total_activation_time = 0.0

        self._frozen_frame: Optional[np.ndarray] = None
        self._freeze_start_time = 0.0
        self._last_spark_time = 0.0

        self._sequence_triggered = False
        self._transform_progress = 0.0

    def update(
        self, activation_state: str, activation_progress: float, delta_time_ms: float
    ) -> float:
        """Update cinematic sequence and return effective intensity."""

        self.phase_time_ms += delta_time_ms
        self.total_activation_time += delta_time_ms

        effective_intensity = 0.0

        if activation_state == "STARE_DETECTED":
            if self.current_phase != ActivationPhase.TENSION_BUILD:
                self._start_tension_build()
            effective_intensity = self._process_tension_build()

        elif activation_state == "ACTIVATING":
            effective_intensity = self._process_activation_progress(activation_progress)

        elif activation_state == "POWER_ACTIVE":
            if self.current_phase != ActivationPhase.SUSTAIN:
                self._start_sustain()
            effective_intency = self._process_sustain()

        elif activation_state in ("IDLE", "COOLDOWN", "INTERRUPTED"):
            effective_intensity = self._process_cooldown()

        else:
            effective_intensity = activation_progress / 100.0

        self._transform_progress = effective_intensity

        return effective_intensity

    def _start_tension_build(self) -> None:
        """Begin tension buildup phase."""
        self.current_phase = ActivationPhase.TENSION_BUILD
        self.phase_time_ms = 0.0

    def _process_tension_build(self) -> float:
        """Process tension buildup with subtle gradual effects."""
        progress = min(1.0, self.phase_time_ms / self.config.tension_duration_ms)

        eased = self._ease_out_quart(progress)

        intensity_range = (
            self.config.tension_max_intensity - self.config.tension_min_intensity
        )
        return self.config.tension_min_intensity + eased * intensity_range

    def _start_freeze_frame(self, frame: np.ndarray) -> None:
        """Trigger freeze-frame effect."""
        if not self.config.enable_freeze_frame:
            return

        self.current_phase = ActivationPhase.FREEZE_IMPACT
        self._frozen_frame = frame.copy()
        self._freeze_start_time = self.phase_time_ms

    def _process_freeze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply freeze-frame effect with chromatic aberration."""
        if self._frozen_frame is None:
            return frame

        elapsed = self.phase_time_ms - self._freeze_start_time
        progress = min(1.0, elapsed / self.config.freeze_duration_ms)

        result = self._frozen_frame.copy()

        intensity = (1.0 - progress) * 0.3

        result = self._apply_chromatic_aberration(result, intensity)

        return result

    def _start_glitch_pulse(self) -> None:
        """Begin glitch pulse effect."""
        self.current_phase = ActivationPhase.GLITCH_PULSE
        self.phase_time_ms = 0.0

    def _process_glitch_pulse(self, frame: np.ndarray) -> np.ndarray:
        """Apply glitch/scanline effect."""
        if not self.config.enable_glitch_pulse:
            return frame

        progress = self.phase_time_ms / self.config.glitch_duration_ms
        if progress > 1.0:
            return frame

        result = frame.copy()

        if progress < 0.5:
            glitch_intensity = self.config.glitch_intensity * (1 - progress * 2)
        else:
            glitch_intensity = self.config.glitch_intensity * ((progress - 0.5) * 2)

        result = self._apply_glitch_effect(result, glitch_intensity)

        return result

    def _start_eye_ignition(self) -> None:
        """Begin eye ignition effect."""
        self.current_phase = ActivationPhase.EYE_IGNITION
        self.phase_time_ms = 0.0
        self._last_spark_time = 0

    def _process_eye_ignition(self, frame: np.ndarray) -> np.ndarray:
        """Apply ignition sparks and energy."""
        if not self.config.enable_ignition_spark:
            return frame

        result = frame.copy()

        progress = self.phase_time_ms / self.config.ignition_duration_ms

        spark_intensity = math.sin(progress * math.pi) * 0.5

        if self.phase_time_ms - self._last_spark_time > 30:
            result = self._add_sparks(result, spark_intensity)
            self._last_spark_time = self.phase_time_ms

        return result

    def _start_sustain(self) -> None:
        """Enter sustained power state."""
        self.current_phase = ActivationPhase.SUSTAIN
        self.phase_time_ms = 0.0

    def _process_sustain(self) -> float:
        """Process sustained power state."""
        return 1.0

    def _process_cooldown(self) -> float:
        """Process cooldown phase."""
        self.current_phase = ActivationPhase.IDLE
        self.phase_time_ms = 0.0
        self._frozen_frame = None
        self._sequence_triggered = False
        return 0.0

    def _process_activation_progress(self, activation_progress: float) -> float:
        """Process the main activation with sequence triggers."""

        if not self._sequence_triggered and activation_progress > 0.0:
            self._sequence_triggered = True

        progress = activation_progress / 100.0

        if self.current_phase == ActivationPhase.TENSION_BUILD:
            if progress > 0.15 and self.config.enable_freeze_frame:
                self.current_phase = ActivationPhase.FREEZE_IMPACT
                self.phase_time_ms = 0.0

        elif self.current_phase == ActivationPhase.FREEZE_IMPACT:
            if self.phase_time_ms > self.config.freeze_duration_ms:
                if self.config.enable_glitch_pulse:
                    self.current_phase = ActivationPhase.GLITCH_PULSE
                    self.phase_time_ms = 0.0

        elif self.current_phase == ActivationPhase.GLITCH_PULSE:
            if self.phase_time_ms > self.config.glitch_duration_ms:
                if self.config.enable_ignition_spark:
                    self.current_phase = ActivationPhase.EYE_IGNITION
                    self.phase_time_ms = 0.0

        elif self.current_phase == ActivationPhase.EYE_IGNITION:
            if self.phase_time_ms > self.config.ignition_duration_ms:
                self.current_phase = ActivationPhase.FULL_TRANSFORM
                self.phase_time_ms = 0.0

        elif self.current_phase == ActivationPhase.FULL_TRANSFORM:
            transform_progress = min(
                1.0, self.phase_time_ms / self.config.transform_ramp_ms
            )
            progress = 0.3 + self._ease_out_cubic(transform_progress) * 0.7

        effective_intensity = progress

        return effective_intensity

    def _apply_chromatic_aberration(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply subtle chromatic aberration effect."""
        h, w = frame.shape[:2]

        shift = int(w * intensity * 0.012)

        result = frame.copy()

        if shift > 0:
            result[:, shift:] = frame[:, :-shift]
            result[:, :shift] = frame[:, w - shift : w]

        return result

    def _apply_glitch_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply digital glitch effect."""
        result = frame.copy()
        h, w = frame.shape[:2]

        num_tears = int(intensity * 8)

        for _ in range(num_tears):
            y = np.random.randint(0, h)
            height = np.random.randint(1, int(h * 0.05))

            offset = np.random.randint(-int(w * 0.1), int(w * 0.1))

            if 0 <= y < h and 0 <= y + height <= h:
                slice_end = min(y + height, h)
                if offset > 0:
                    result[y:slice_end, offset:] = frame[y:slice_end, :-offset]
                elif offset < 0:
                    result[y:slice_end, :offset] = frame[y:slice_end, -offset:]

        scanline_intensity = intensity * 0.1
        for y in range(0, h, 4):
            result[y : y + 1, :] = np.clip(
                result[y : y + 1, :].astype(np.int16) * (1 + scanline_intensity), 0, 255
            ).astype(np.uint8)

        return result

    def _add_sparks(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add ignition sparks to frame."""
        result = frame.copy()
        h, w = frame.shape[:2]

        num_sparks = int(self.config.spark_count * intensity)

        for _ in range(num_sparks):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)

            length = np.random.randint(3, 15)
            angle = np.random.uniform(0, 2 * math.pi)

            end_x = int(x + math.cos(angle) * length)
            end_y = int(y + math.sin(angle) * length)

            if 0 <= end_x < w and 0 <= end_y < h:
                color = (
                    min(255, int(255 * intensity)),
                    min(255, int(180 * intensity)),
                    min(255, int(80 * intensity)),
                )
                cv2.line(result, (x, y), (end_x, end_y), color, 1, cv2.LINE_AA)

        return result

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """Ease out cubic easing function."""
        return 1 - math.pow(1 - t, 3)

    @staticmethod
    def _ease_out_quart(t: float) -> float:
        """Ease out quart - smoother, more gradual."""
        return 1 - math.pow(1 - t, 4)

    def get_phase(self) -> ActivationPhase:
        """Get current activation phase."""
        return self.current_phase

    def get_transform_progress(self) -> float:
        """Get current transformation progress (0-1)."""
        return self._transform_progress

    def is_sequence_active(self) -> bool:
        """Check if cinematic sequence is active."""
        return self.current_phase not in (
            ActivationPhase.IDLE,
            ActivationPhase.COOLDOWN,
        )

    def apply_cinematic_effects(self, frame: np.ndarray) -> np.ndarray:
        """Apply any active cinematic post-processing to frame."""

        if self.current_phase == ActivationPhase.FREEZE_IMPACT:
            return self._process_freeze_frame(frame)

        elif self.current_phase == ActivationPhase.GLITCH_PULSE:
            return self._process_glitch_pulse(frame)

        elif self.current_phase == ActivationPhase.EYE_IGNITION:
            return self._process_eye_ignition(frame)

        return frame


class CinematicGrading:
    """Professional color grading for cinematic effect."""

    def __init__(self):
        self.contrast = 1.15
        self.saturation = 1.1
        self.red_shift = 8
        self.shadow_gain = 1.1

    def apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply cinematic color grading."""
        if intensity < 0.1:
            return frame

        result = frame.astype(np.float32)

        result = self._apply_contrast(result, self.contrast)

        result = self._apply_saturation(result, self.saturation)

        result = self._apply_red_shift(result, self.red_shift * intensity)

        result = self._apply_vignette(result, intensity * 0.3)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_contrast(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Apply contrast adjustment."""
        mean = np.mean(frame)
        return mean + (frame - mean) * factor

    def _apply_saturation(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Apply saturation adjustment."""
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        return frame * factor + gray[:, :, np.newaxis] * (1 - factor)

    def _apply_red_shift(self, frame: np.ndarray, shift: float) -> np.ndarray:
        """Add red shift to shadows for supernatural feel."""
        result = frame.copy()
        result[:, :, 0] += shift
        result[:, :, 1] += shift * 0.3
        return result

    def _apply_vignette(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply subtle vignette."""
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]

        center_y, center_x = h / 2, w / 2

        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        vignette = 1 - (dist / max_dist) * intensity
        vignette = np.clip(vignette, 0, 1)

        return frame * vignette[:, :, np.newaxis]
