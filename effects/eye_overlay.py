"""Dynamic anime-inspired eye overlay tied to face landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext


@dataclass
class EyePose:
    center: Tuple[float, float] = (0.0, 0.0)
    axes: Tuple[float, float] = (20.0, 12.0)
    angle: float = 0.0
    intensity: float = 0.0


class EyeOverlayEffect(BaseEffect):
    """Draws stylized eye auras and iris rings using live landmarks."""

    name = "eye_overlay"
    priority = 20
    minimum_quality = 1

    def __init__(self, enabled: bool = True, glow_intensity: float = 0.85):
        super().__init__(enabled=enabled)
        self.glow_intensity = max(0.0, min(1.5, glow_intensity))
        self.current_intensity = 0.0
        self._pose_cache: Dict[str, EyePose] = {
            "left": EyePose(),
            "right": EyePose(),
        }

    def update(self, context: EffectContext) -> None:
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        contact_bonus = 0.0
        if (
            context.tracking_data is not None
            and context.tracking_data.eye_contact.in_contact
        ):
            contact_bonus = 0.2
        self.current_intensity = min(
            1.0, (progress * 0.8 + contact_bonus) * self.glow_intensity
        )

    def process(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if (
            not self.enabled
            or context.tracking_data is None
            or context.tracking_data.face is None
        ):
            return frame

        tracking_data = context.tracking_data
        if not tracking_data.face.detected:
            return frame

        overlay = frame.copy()

        # Draw on overlay
        self._draw_eye(overlay, tracking_data.left_eye.landmarks, context, "left")
        self._draw_eye(overlay, tracking_data.right_eye.landmarks, context, "right")

        # Blend with alpha so it looks like a real contact lens
        return BaseEffect.alpha_blend(frame, overlay, 0.65)

    def _draw_eye(
        self, overlay: np.ndarray, landmarks, context: EffectContext, side: str
    ) -> None:
        if not landmarks:
            return

        points = np.asarray(landmarks, dtype=np.int32)
        if points.shape[0] < 3:
            return

        # Haar mock landmarks collapse many indices into the center point.
        # MediaPipe has distinct coordinates for all 16 points.
        unique_points = np.unique(points, axis=0)
        is_high_fidelity = unique_points.shape[0] >= 10

        xs = points[:, 0]
        ys = points[:, 1]

        # Better center calculation for high fidelity
        if is_high_fidelity:
            M = cv2.moments(points)
            if M["m00"] == 0:
                return
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

        eye_width = np.max(xs) - np.min(xs)
        if eye_width < 10:
            return

        iris_radius = int(eye_width * 0.38)

        # ==========================================
        # 1. DRAW PROCEDURAL SHARINGAN
        # ==========================================
        patch_size = iris_radius * 3
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        pc = patch_size // 2

        # Reduced vividness/brightness to match ambient low-light
        red_color = (15, 15, 140)
        cv2.circle(patch, (pc, pc), iris_radius, red_color, -1, cv2.LINE_AA)
        cv2.circle(patch, (pc, pc), iris_radius, (0, 0, 0), 2, cv2.LINE_AA)

        inner_radius = int(iris_radius * 0.55)
        cv2.circle(patch, (pc, pc), inner_radius, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(patch, (pc, pc), int(iris_radius * 0.20), (0, 0, 0), -1, cv2.LINE_AA)

        tomoe_radius = max(2, int(iris_radius * 0.15))
        rotation = context.timestamp_ms / 300.0

        for i in range(3):
            angle = rotation + (i * 2 * math.pi / 3)
            tx = int(pc + math.cos(angle) * inner_radius)
            ty = int(pc + math.sin(angle) * inner_radius)
            cv2.circle(patch, (tx, ty), tomoe_radius, (0, 0, 0), -1, cv2.LINE_AA)

            tail_angle = angle + math.pi / 2.5
            tail_len = tomoe_radius * 2.5
            pts = np.array(
                [
                    (tx, ty),
                    (
                        int(tx + math.cos(tail_angle - 0.2) * tail_len),
                        int(ty + math.sin(tail_angle - 0.2) * tail_len),
                    ),
                    (
                        int(tx + math.cos(tail_angle + 0.8) * (tail_len * 0.4)),
                        int(ty + math.sin(tail_angle + 0.8) * (tail_len * 0.4)),
                    ),
                ],
                np.int32,
            )
            cv2.fillPoly(patch, [pts], (0, 0, 0), cv2.LINE_AA)

        # Cast a soft shadow from the upper eyelid onto the top portion of the pattern
        shadow_gradient = (
            np.linspace(0.4, 1.0, patch_size)
            .reshape(patch_size, 1, 1)
            .astype(np.float32)
        )
        patch = (patch.astype(np.float32) * shadow_gradient).astype(np.uint8)

        # ==========================================
        # 2. EYELID MASKING & BLENDING
        # ==========================================
        # Place patch into full frame canvas
        canvas = np.zeros_like(overlay)
        x_min = cx - pc
        x_max = cx + patch_size - pc
        y_min = cy - pc
        y_max = cy + patch_size - pc

        # Calculate safe bounds for canvas and patch
        c_xmin, c_xmax = max(0, x_min), min(overlay.shape[1], x_max)
        c_ymin, c_ymax = max(0, y_min), min(overlay.shape[0], y_max)

        p_xmin = c_xmin - x_min
        p_xmax = patch_size - (x_max - c_xmax)
        p_ymin = c_ymin - y_min
        p_ymax = patch_size - (y_max - c_ymax)

        if c_xmax > c_xmin and c_ymax > c_ymin:
            canvas[c_ymin:c_ymax, c_xmin:c_xmax] = patch[p_ymin:p_ymax, p_xmin:p_xmax]

        # The eyelid mask (restricts drawing to inside the eyelids)
        eyelid_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)

        if is_high_fidelity:
            # PERFECT MESH MASKING
            cv2.fillPoly(eyelid_mask, [points], 255)
        else:
            # HAAR FALLBACK MASKING
            eye_height = int(eye_width * 0.48)
            cv2.ellipse(
                eyelid_mask,
                (cx, cy),
                (int(eye_width // 2), int(eye_height // 2)),
                0,
                0,
                360,
                255,
                -1,
            )

        # The Sharingan mask (restricts drawing to the Sharingan circle)
        sharingan_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
        cv2.circle(sharingan_mask, (cx, cy), iris_radius, 255, -1, cv2.LINE_AA)

        # Final mask is the intersection: Sharingan shape properly clipped by the eyelids
        final_mask = cv2.bitwise_and(sharingan_mask, eyelid_mask)

        # ==========================================
        # 3. HYPER-REALISTIC FALLBACK BLENDING
        # ==========================================
        # Feather the edges for smooth integration into the iris
        mask_float = (
            cv2.GaussianBlur(final_mask, (11, 11), 0).astype(np.float32) / 255.0
        )

        # Max opacity 0.65 to let ambient shadows and room lighting show through
        mask_float *= 0.65
        mask_3d = np.dstack([mask_float] * 3)

        # Preserve specular highlights (white light reflections on the cornea)
        gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(gray_overlay, 210, 255, cv2.THRESH_BINARY)

        # Remove Sharingan opacity over natural highlights
        highlight_mask_float = highlight_mask.astype(np.float32) / 255.0
        mask_3d = mask_3d * (1.0 - np.dstack([highlight_mask_float] * 3))

        # Apply the final photorealistic blend
        overlay_float = overlay.astype(np.float32)
        canvas_float = canvas.astype(np.float32)

        overlay[:] = (overlay_float * (1.0 - mask_3d) + canvas_float * mask_3d).astype(
            np.uint8
        )
