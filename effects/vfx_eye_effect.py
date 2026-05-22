"""
Professional VFX Eye Transformation Effect
Complete cinematic eye replacement with 3D perspective, sclera transformation, and activation sequence.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from effects.ai_eye_compositor import AIEyeCompositor, create_ai_compositor
from effects.base_effect import BaseEffect
from effects.cinematic_activation_system import (
    CinematicActivationSystem,
    CinematicGrading,
    CinematicSequenceConfig,
)
from effects.effect_data import EffectContext
from effects.gpu_eye_compositor import GPUEyeCompositor, create_gpu_compositor
from effects.realistic_eye_compositor import (
    ProfessionalEyeCompositor,
    create_professional_compositor,
    create_webcam_enhancer,
)
from tracking.ai_eye_segmentation import (
    AIEyeSegmenter,
    BlinkDetector,
    EyeGeometryEstimator,
    GazeEstimator,
)
from tracking.enhanced_eye_tracker import EnhancedEyeTracker
from tracking.smoothed_tracking import EyeTrackerStabilizer


class VFXEyeEffect(BaseEffect):
    """
    Professional VFX eye transformation effect.
    Replaces eyes with Sharingan using proper 3D perspective and cinematic compositing.
    """

    name = "vfx_eye"
    priority = 15
    minimum_quality = 1

    def __init__(self, enabled: bool = True, intensity: float = 1.0):
        super().__init__(enabled=enabled)
        self.intensity = intensity

        from config.assets import get_sharingan_pack_path

        pack_path = get_sharingan_pack_path()
        self.compositor = create_professional_compositor(pack_path)
        self.ai_compositor = create_ai_compositor(pack_path)

        try:
            self.gpu_compositor = create_gpu_compositor(pack_path)
            self._use_gpu = True
        except:
            self.gpu_compositor = None
            self._use_gpu = False

        self.webcam_enhancer = create_webcam_enhancer()

        self.eye_tracker = EnhancedEyeTracker()
        self.stabilizer = EyeTrackerStabilizer()

        self.ai_segmenter = AIEyeSegmenter()
        self.gaze_estimator = GazeEstimator()
        self.geometry_estimator = EyeGeometryEstimator()
        self.blink_detector = BlinkDetector()

        self.cinematic_config = CinematicSequenceConfig()
        self.cinematic_system = CinematicActivationSystem(self.cinematic_config)

        self.cinematic_grading = CinematicGrading()

        self._smoothing_factor = 0.4
        self._last_left_mesh = None
        self._last_right_mesh = None

        self._tracking_confidence = 0.0
        self._current_intensity = 0.0

        self._webcam_enhanced_frame = None
        self._last_intensity = 0.0

        self._use_ai_compositing = True

    def update(self, context: EffectContext) -> None:
        """Update effect state for current frame."""
        activation_intensity = 0.0

        if context.activation_data:
            activation_state = context.activation_data.state.value
            activation_progress = context.activation_data.activation_progress

            delta_time = context.delta_ms
            activation_intensity = self.cinematic_system.update(
                activation_state, activation_progress, delta_time
            )

            effect_strength = getattr(self, "intensity", 1.0) if self.enabled else 0.0
            self._current_intensity = activation_intensity * effect_strength

        if context.tracking_data:
            self._tracking_confidence = context.tracking_data.overall_confidence

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        """Process frame with VFX eye transformation."""
        if not self.enabled:
            return frame

        if context.tracking_data is None:
            return frame

        tracking = context.tracking_data
        if not tracking.face or not tracking.face.detected:
            return frame

        if not tracking.face.landmarks or len(tracking.face.landmarks) < 478:
            return frame

        landmarks = tracking.face.landmarks

        intensity = self._current_intensity * self._get_confidence_multiplier()

        if intensity < 0.05:
            if self.cinematic_system.is_sequence_active():
                pass
            else:
                return frame

        result = frame.copy()

        left_mesh = self.eye_tracker.extract_eye_mesh(landmarks, "left")
        right_mesh = self.eye_tracker.extract_eye_mesh(landmarks, "right")

        confidence = self._tracking_confidence

        left_eye_data = {"detected": False}
        right_eye_data = {"detected": False}

        left_seg = None
        right_seg = None
        left_gaze = None
        right_gaze = None
        left_geom = None
        right_geom = None
        left_blink = None
        right_blink = None

        if self._use_ai_compositing and confidence > 0.5:
            left_seg = self.ai_segmenter.segment_eye(frame, landmarks, "left")
            right_seg = self.ai_segmenter.segment_eye(frame, landmarks, "right")

            left_gaze = self.gaze_estimator.estimate_gaze(landmarks, "left")
            right_gaze = self.gaze_estimator.estimate_gaze(landmarks, "right")

            left_geom = self.geometry_estimator.estimate_geometry(landmarks, "left")
            right_geom = self.geometry_estimator.estimate_geometry(landmarks, "right")

            left_blink = self.blink_detector.detect_blink(landmarks, "left")
            right_blink = self.blink_detector.detect_blink(landmarks, "right")

        self.stabilizer.set_time(context.time_ms / 1000.0)

        left_iris_conf = left_mesh.confidence if left_mesh else 0.0
        right_iris_conf = right_mesh.confidence if right_mesh else 0.0

        mesh_conf_left = confidence * 0.5 + left_iris_conf * 0.5
        mesh_conf_right = confidence * 0.5 + right_iris_conf * 0.5

        if left_mesh and mesh_conf_left > 0.35:
            if self._last_left_mesh:
                alpha = 0.55
                left_mesh.iris_center = (
                    self._last_left_mesh.iris_center[0] * (1 - alpha)
                    + left_mesh.iris_center[0] * alpha,
                    self._last_left_mesh.iris_center[1] * (1 - alpha)
                    + left_mesh.iris_center[1] * alpha,
                )
            self._last_left_mesh = left_mesh

            left_eye_data = {
                "detected": True,
                "center": left_mesh.iris_center,
                "iris_center": left_mesh.iris_center,
                "width": left_mesh.eye_width,
                "upper_lid": left_mesh.upper_eyelid,
                "lower_lid": left_mesh.lower_eyelid,
                "angle": left_mesh.eye_angle,
                "tilt": left_mesh.eye_tilt,
                "openness": left_mesh.openness,
            }

        if right_mesh and mesh_conf_right > 0.35:
            if self._last_right_mesh:
                alpha = 0.55
                right_mesh.iris_center = (
                    self._last_right_mesh.iris_center[0] * (1 - alpha)
                    + right_mesh.iris_center[0] * alpha,
                    self._last_right_mesh.iris_center[1] * (1 - alpha)
                    + right_mesh.iris_center[1] * alpha,
                )
            self._last_right_mesh = right_mesh

        if self._use_gpu and self.gpu_compositor:
            result = self.gpu_compositor.composite_eyes(
                result, left_eye_data, right_eye_data, landmarks, intensity
            )
        elif self._use_ai_compositing and left_seg is not None and confidence > 0.5:
            result = self.ai_compositor.composite_eyes(
                result,
                left_seg,
                right_seg,
                left_gaze,
                right_gaze,
                left_geom,
                right_geom,
                left_blink,
                right_blink,
                landmarks,
                intensity,
            )
        else:
            result = self.compositor.composite_eyes(
                result, left_eye_data, right_eye_data, landmarks, intensity
            )

        if intensity > 0.2:
            result = self.compositor.apply_cinematic_grading(result, intensity * 0.5)

        if self.cinematic_system.is_sequence_active():
            result = self.cinematic_system.apply_cinematic_effects(result)

        if intensity > 0.3:
            result = self.cinematic_grading.apply_cinematic_grading(result, intensity)

        return result

    def _smooth_eye_mesh(self, current, last):
        """Apply temporal smoothing to eye mesh."""
        if last is None:
            return current

        alpha = self._smoothing_factor

        smoothed_mesh = type(current)(
            iris_center=(
                last.iris_center[0] * (1 - alpha) + current.iris_center[0] * alpha,
                last.iris_center[1] * (1 - alpha) + current.iris_center[1] * alpha,
            ),
            iris_radius=last.iris_radius * (1 - alpha) + current.iris_radius * alpha,
            eye_center=(
                last.eye_center[0] * (1 - alpha) + current.eye_center[0] * alpha,
                last.eye_center[1] * (1 - alpha) + current.eye_center[1] * alpha,
            ),
            eye_width=last.eye_width * (1 - alpha) + current.eye_width * alpha,
            eye_height=last.eye_height * (1 - alpha) + current.eye_height * alpha,
            upper_eyelid=[
                (
                    (
                        last.upper_eyelid[i][0] * (1 - alpha)
                        + current.upper_eyelid[i][0] * alpha
                        if i < len(last.upper_eyelid) and i < len(current.upper_eyelid)
                        else current.upper_eyelid[i][0]
                    ),
                    (
                        last.upper_eyelid[i][1] * (1 - alpha)
                        + current.upper_eyelid[i][1] * alpha
                        if i < len(last.upper_eyelid) and i < len(current.upper_eyelid)
                        else current.upper_eyelid[i][1]
                    ),
                )
                for i in range(len(current.upper_eyelid))
            ],
            lower_eyelid=[
                (
                    (
                        last.lower_eyelid[i][0] * (1 - alpha)
                        + current.lower_eyelid[i][0] * alpha
                        if i < len(last.lower_eyelid) and i < len(current.lower_eyelid)
                        else current.lower_eyelid[i][0]
                    ),
                    (
                        last.lower_eyelid[i][1] * (1 - alpha)
                        + current.lower_eyelid[i][1] * alpha
                        if i < len(last.lower_eyelid) and i < len(current.lower_eyelid)
                        else current.lower_eyelid[i][1]
                    ),
                )
                for i in range(len(current.lower_eyelid))
            ],
            eye_corners=current.eye_corners,
            eye_angle=last.eye_angle * (1 - alpha) + current.eye_angle * alpha,
            eye_tilt=last.eye_tilt * (1 - alpha) + current.eye_tilt * alpha,
            openness=last.openness * (1 - alpha) + current.openness * alpha,
            blink_state=last.blink_state * (1 - alpha) + current.blink_state * alpha,
        )

        return smoothed_mesh

    def _get_confidence_multiplier(self) -> float:
        """Get intensity multiplier based on tracking confidence."""
        base = 0.5
        return base + (self._tracking_confidence * 0.5)

    def handle_event(self, event_name: str, payload: dict) -> None:
        """Handle activation events for cinematic sequence triggers."""
        if event_name == "on_stare_start":
            pass

        elif event_name == "on_activation_start":
            pass

        elif event_name == "on_activation_complete":
            self.cinematic_system._sequence_triggered = False

        elif event_name == "on_interrupt":
            self.cinematic_system._sequence_triggered = False

        elif event_name == "on_cooldown_start":
            pass

        elif event_name == "on_cooldown_end":
            pass

    def release(self) -> None:
        """Release resources."""
        self._last_left_mesh = None
        self._last_right_mesh = None


class ImageQualityEnhancer:
    """Enhance base image quality before VFX compositing."""

    def __init__(self):
        self.sharpness = 1.3
        self.denoise_strength = 3
        self.local_contrast = 0.8

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply quality enhancements to frame."""
        result = frame.copy()

        result = cv2.fastNlMeansDenoisingColored(
            result, None, self.denoise_strength, self.denoise_strength, 7, 21
        )

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel * (self.sharpness - 1))
        result = cv2.addWeighted(result, 1.0, sharpened, self.sharpness - 1, 0)

        result = self._apply_local_contrast(result)

        result = self._auto_levels(result)

        return result

    def _apply_local_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply local contrast enhancement (unsharp mask style)."""
        blurred = cv2.GaussianBlur(frame, (0, 0), 10)

        result = cv2.addWeighted(
            frame, 1.0 + self.local_contrast, blurred, -self.local_contrast, 0
        )
        return result

    def _auto_levels(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic levels for better dynamic range."""
        for c in range(3):
            channel = frame[:, :, c]

            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

            cumsum = np.cumsum(hist)
            total = cumsum[-1]

            low = np.searchsorted(cumsum, total * 0.02)
            high = np.searchsorted(cumsum, total * 0.98)

            if high > low:
                channel = ((channel - low) * 255 / (high - low)).astype(np.uint8)
                channel = np.clip(channel, 0, 255)

            frame[:, :, c] = channel

        return frame


def create_professional_eye_effect() -> VFXEyeEffect:
    """Factory function to create the professional eye effect."""
    return VFXEyeEffect(enabled=True, intensity=1.0)
