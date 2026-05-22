"""
AI-Driven Eye Compositor
Uses eye segmentation masks for realistic Sharingan integration.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


class AIEyeCompositor:
    """
    AI-driven eye compositor for cinematic Sharingan transformation.
    Uses per-pixel eye segmentation for realistic integration.
    """

    def __init__(self, pack_path: Optional[str] = None):
        if pack_path is None:
            from config.assets import get_sharingan_pack_path

            pack_path = get_sharingan_pack_path()

        self.pack_path = pack_path
        self.textures = self._load_textures()
        self._select_random_texture()

        self._last_left_center = None
        self._last_right_center = None

        self._time = 0.0
        self._left_texture = None
        self._right_texture = None

    def _load_textures(self) -> List[np.ndarray]:
        """Load realistic eye textures from Sharingan Pack."""
        textures = []

        realistic_dir = os.path.join(self.pack_path, "realistic_eyes")
        if os.path.exists(realistic_dir):
            for fname in os.listdir(realistic_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img = cv2.imread(
                        os.path.join(realistic_dir, fname), cv2.IMREAD_UNCHANGED
                    )
                    if img is not None:
                        textures.append(img)

        if not textures:
            renders_dir = os.path.join(self.pack_path, "Renders")
            if os.path.exists(renders_dir):
                for fname in os.listdir(renders_dir):
                    if fname.endswith(".png"):
                        img = cv2.imread(
                            os.path.join(renders_dir, fname), cv2.IMREAD_UNCHANGED
                        )
                        if img is not None:
                            textures.append(img)

        return textures

    def _select_random_texture(self):
        """Select random eye texture."""
        if self.textures:
            self.current_texture = self.textures[
                np.random.randint(0, len(self.textures))
            ]

    def composite_eyes(
        self,
        frame: np.ndarray,
        left_segmentation,
        right_segmentation,
        left_gaze,
        right_gaze,
        left_geometry,
        right_geometry,
        left_blink,
        right_blink,
        landmarks: List,
        intensity: float,
    ) -> np.ndarray:
        """Composite eyes with AI-driven segmentation."""

        if intensity < 0.05:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        eye_centers = []

        if left_segmentation is not None:
            result = self._composite_single_eye(
                result,
                left_segmentation,
                left_gaze,
                left_geometry,
                left_blink,
                landmarks,
                intensity,
                "left",
                h,
                w,
            )
            if left_gaze:
                eye_centers.append(left_gaze.iris_offset)

        if right_segmentation is not None:
            result = self._composite_single_eye(
                result,
                right_segmentation,
                right_gaze,
                right_geometry,
                right_blink,
                landmarks,
                intensity,
                "right",
                h,
                w,
            )
            if right_gaze:
                eye_centers.append(right_gaze.iris_offset)

        if intensity > 0.15:
            result = self._apply_cinematic_grading(result, intensity)

        if intensity > 0.15:
            result = self._apply_restraint(result, intensity)

        self._time += 0.016

        return result

    def _composite_single_eye(
        self,
        frame: np.ndarray,
        segmentation,
        gaze,
        geometry,
        blink,
        landmarks,
        intensity: float,
        eye_side: str,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Composite single eye with full AI segmentation."""

        eye_region = segmentation.eye_region
        iris_mask = segmentation.iris_mask
        upper_lid = segmentation.upper_lid_mask
        lower_lid = segmentation.lower_lid_mask

        ys, xs = np.where(eye_region > 0)
        if len(xs) == 0:
            return frame

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w < 10 or roi_h < 10:
            return frame

        roi = frame[y1:y2, x1:x2].copy()

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_threshold = np.percentile(roi_gray[eye_region > 0], 20)
        dark_mask = (roi_gray < dark_threshold).astype(np.float32) * 0.4

        tinted = roi.astype(np.float32)
        tinted = tinted * (1.0 - 0.25 * intensity)
        tinted[:, :, 0] += 20 * intensity
        tinted[:, :, 1] += 8 * intensity
        tinted[:, :, 2] += 2 * intensity
        frame[y1:y2, x1:x2] = np.clip(tinted, 0, 255).astype(np.uint8)

        texture, tex_alpha = self._prepare_texture(roi_h, roi_w, gaze, geometry)

        if texture is None:
            return frame

        if gaze:
            texture = self._apply_gaze_deformation(texture, gaze, geometry)

        blink_openness = blink.get("openness", 1.0) if blink else 1.0
        texture, tex_alpha = self._apply_blink_deformation(
            texture, tex_alpha, blink_openness
        )

        texture = self._apply_curvature_warp(texture, geometry)

        tex_h, tex_w = texture.shape[:2]
        if tex_w != roi_w or tex_h != roi_h:
            texture = cv2.resize(texture, (roi_w, roi_h))
            tex_alpha = cv2.resize(tex_alpha, (roi_w, roi_h))

        local_eye_region = cv2.resize(eye_region, (roi_w, roi_h))
        local_upper_lid = cv2.resize(upper_lid, (roi_w, roi_h))
        local_lower_lid = cv2.resize(lower_lid, (roi_w, roi_h))

        lid_mask = np.maximum(local_upper_lid, local_lower_lid)

        eye_fill = local_eye_region.copy()
        eye_fill = cv2.GaussianBlur(eye_fill, (5, 5), 0)

        combined_mask = np.maximum(tex_alpha * 0.85, eye_fill * 0.7)

        combined_mask = combined_mask * (1 - lid_mask * 0.4)

        bg = frame[y1:y2, x1:x2].astype(np.float32)

        dark_eye_bg = bg * 0.5
        dark_eye_bg[:, :, 0] += 10
        dark_eye_bg[:, :, 1] += 3

        bg_blend = local_eye_region[:, :, np.newaxis] / 255.0
        bg = dark_eye_bg * bg_blend + bg * (1 - bg_blend)

        fg = texture.astype(np.float32)

        fg_matched = self._match_skin_tone(fg, bg, combined_mask)

        alpha_3d = np.dstack([combined_mask, combined_mask, combined_mask])

        blended = (bg * (1 - alpha_3d) + fg_matched * alpha_3d).astype(np.uint8)

        if hasattr(self, "_time"):
            blended = self._add_corneal_reflection(
                blended, (roi_w // 2, roi_h // 2), self._time * 1000, eye_side
            )

        blended = self._add_sclera_realism(blended, (roi_w // 2, roi_h // 2), roi_w)

        upper_lid_occlusion = cv2.resize(upper_lid, (roi_w, roi_h))

        lid_compress = (
            int(roi_h * (1 - blink_openness) * 0.2) if blink_openness < 1.0 else 0
        )
        if lid_compress > 0:
            upper_lid_occlusion[:lid_compress, :] = 1.0
            upper_lid_occlusion = cv2.GaussianBlur(upper_lid_occlusion, (9, 9), 0)

        blended = blended * (1 - upper_lid_occlusion[:, :, np.newaxis] * 0.8)

        blended = self._refine_eyelid_integration(
            blended, local_upper_lid, local_lower_lid, blink_openness
        )

        blended = self._add_eyelid_shadow_contamination(
            blended, local_upper_lid, local_lower_lid
        )

        blended = self._soften_eye_socket_shadow(
            blended, (roi_w // 2, roi_h // 2), roi_w
        )

        blended = self._inherit_face_lighting(blended, roi, local_eye_region)

        if local_eye_region is not None:
            blended = self._blend_with_webcam_noise(blended, roi, local_eye_region)

        frame[y1:y2, x1:x2] = blended

        return frame

    def _prepare_texture(
        self, target_h: int, target_w: int, gaze, geometry
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare texture with geometry-aware scaling."""
        if self.current_texture is None:
            return None, None

        tex = self.current_texture

        scale = 1.0
        if geometry:
            scale = 1.0 + (geometry.compression_x - 1.0) * 0.3

        new_w = int(target_w * scale)
        new_h = target_h

        if tex.shape[2] == 4:
            bgra = cv2.resize(tex, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            color = bgra[:, :, :3]
            alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        else:
            color = cv2.resize(tex, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            alpha = np.ones((new_h, new_w), dtype=np.float32)

        if new_w != target_w:
            offset = (new_w - target_w) // 2
            color = color[:, offset : offset + target_w]
            alpha = alpha[:, offset : offset + target_w]

        return color, alpha

    def _apply_gaze_deformation(
        self, texture: np.ndarray, gaze, geometry
    ) -> np.ndarray:
        """Apply gaze-based iris deformation."""
        if not gaze or not hasattr(gaze, "iris_offset"):
            return texture

        offset_x, offset_y = gaze.iris_offset

        h, w = texture.shape[:2]

        shift_x = int(offset_x * w * 0.15)
        shift_y = int(offset_y * h * 0.1)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        deformed = cv2.warpAffine(
            texture, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        return deformed

    def _apply_blink_deformation(
        self, texture: np.ndarray, alpha: np.ndarray, openness: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply blink-driven texture compression with smooth transitions."""
        if openness >= 0.95:
            return texture, alpha
        if openness < 0.05:
            return np.zeros_like(texture), np.zeros_like(alpha)

        h, w = texture.shape[:2]

        compression = 1.0 - (1.0 - openness) * 0.25
        new_h = int(h * compression)

        compressed = cv2.resize(texture, (w, new_h))

        if new_h < h:
            padded = np.zeros((h, w, 3), dtype=texture.dtype)
            offset = (h - new_h) // 2
            padded[offset : offset + new_h] = compressed
            compressed = padded

        alpha_compressed = cv2.resize(alpha, (w, new_h))
        if new_h < h:
            padded_alpha = np.zeros((h, w), dtype=alpha.dtype)
            padded_alpha[offset : offset + new_h] = alpha_compressed
            alpha_compressed = padded_alpha

        lid_occlusion = np.zeros((h, w), dtype=np.float32)

        lid_height = int((1 - openness) * h * 0.35)
        if lid_height > 0:
            for y in range(lid_height):
                falloff = (y / lid_height) ** 2
                lid_occlusion[y, :] = falloff * (1 - openness)

        lid_occlusion = cv2.GaussianBlur(lid_occlusion, (13, 13), 0)

        final_alpha = alpha_compressed * (1 - lid_occlusion)

        darkened = (
            compressed.astype(np.float32) * (1 - lid_occlusion[:, :, np.newaxis] * 0.3)
        ).astype(np.uint8)

        return darkened, final_alpha

    def _apply_curvature_warp(self, texture: np.ndarray, geometry) -> np.ndarray:
        """Apply eyeball curvature warping for perspective realism."""
        if not geometry:
            return texture

        h, w = texture.shape[:2]
        cx, cy = w // 2, h // 2

        curvature = getattr(geometry, "curvature_factor", 1.0)
        compression_x = getattr(geometry, "compression_x", 1.0)
        compression_y = getattr(geometry, "compression_y", 1.0)

        y_grid, x_grid = np.ogrid[:h, :w]

        dx = (x_grid - cx) / cx
        dy = (y_grid - cy) / cy

        dist = np.sqrt(dx**2 + dy**2)

        bulge = 1.0 + 0.1 * (1 - dist**2)
        bulge = np.clip(bulge, 0.9, 1.1)

        warped_x = cx + dx * compression_x * bulge * cx
        warped_y = cy + dy * compression_y * bulge * cy

        warped_x = np.clip(warped_x, 0, w - 1)
        warped_y = np.clip(warped_y, 0, h - 1)

        warped = cv2.remap(
            texture,
            warped_x.astype(np.float32),
            warped_y.astype(np.float32),
            cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return warped

    def _match_skin_tone(
        self, eye_region: np.ndarray, skin_region: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Match eye region color to surrounding skin."""
        if eye_region.size == 0 or skin_region.size == 0:
            return eye_region

        valid_skin = skin_region[mask < 0.3]
        valid_eye = eye_region[mask > 0.5]

        if len(valid_skin) < 100 or len(valid_eye) < 50:
            return eye_region

        skin_mean = np.mean(valid_skin, axis=0)
        eye_mean = np.mean(valid_eye, axis=0)

        ratio = (skin_mean + 1e-6) / (eye_mean + 1e-6)

        matched = eye_region.astype(np.float32) * ratio
        matched = np.clip(matched, 0, 255).astype(np.uint8)

        return cv2.addWeighted(eye_region, 0.3, matched, 0.7, 0)

    def _apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply subtle cinematic color grading."""
        if intensity < 0.1:
            return frame

        result = frame.astype(np.float32)

        result = result * (0.95 + 0.05 * intensity)

        result[:, :, 0] += 3 * intensity
        result[:, :, 1] += 1 * intensity

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _add_corneal_reflection(
        self,
        eye_region: np.ndarray,
        iris_center: Tuple[int, int],
        time_ms: float,
        eye_side: str,
    ) -> np.ndarray:
        """Add realistic wet-eye specular reflection."""
        h, w = eye_region.shape[:2]
        cx, cy = iris_center

        if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
            return eye_region

        phase_offset = 0.3 if eye_side == "left" else 0.7
        breathe = np.sin(time_ms * 0.001 + phase_offset) * 0.5 + 0.5

        offset_x = int(np.sin(time_ms * 0.0008 + phase_offset) * 3)
        offset_y = int(np.cos(time_ms * 0.0006) * 2)

        spec_x = cx + offset_x - 8
        spec_y = cy + offset_y - 6

        if spec_x < 3 or spec_x > w - 3 or spec_y < 3 or spec_y > h - 3:
            return eye_region

        reflection = np.zeros_like(eye_region)

        cv2.ellipse(
            reflection, (spec_x, spec_y), (6, 4), -20, 0, 360, (255, 255, 255), -1
        )

        reflection = cv2.GaussianBlur(reflection, (7, 7), 0)

        reflection_strength = 0.09 + 0.04 * breathe

        result = eye_region.astype(np.float32)
        result = result + reflection * reflection_strength

        second_spec_x = spec_x + 12
        second_spec_y = spec_y + 3
        if second_spec_x < w - 3 and second_spec_y < h - 3:
            second_reflection = np.zeros_like(eye_region)
            cv2.ellipse(
                second_reflection,
                (second_spec_x, second_spec_y),
                (3, 2),
                -10,
                0,
                360,
                (255, 255, 255),
                -1,
            )
            second_reflection = cv2.GaussianBlur(second_reflection, (5, 5), 0)
            result = result + second_reflection * 0.025

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_sclera_realism(
        self, eye_region: np.ndarray, iris_center: Tuple[int, int], eye_width: int
    ) -> np.ndarray:
        """Add realistic sclera with subtle gray gradient and shadow."""
        h, w = eye_region.shape[:2]
        cx, cy = int(iris_center[0]), int(iris_center[1])

        if cx <= 0 or cy <= 0:
            return eye_region

        result = eye_region.astype(np.float32)

        y_grid, x_grid = np.ogrid[:h, :w]

        dist_x = np.abs(x_grid - cx) / (w / 2 + 1)
        dist_y = np.abs(y_grid - cy) / (h / 2 + 1)

        radial_dist = np.sqrt(dist_x**2 + dist_y**2)

        vignette = 1.0 - radial_dist * 0.08
        vignette = np.clip(vignette, 0.92, 1.0)

        result = result * vignette

        shadow_offset_x = int(eye_width * 0.1)

        inner_shadow = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            inner_shadow,
            (cx - shadow_offset_x, cy),
            (int(eye_width * 0.3), int(h * 0.35)),
            0,
            0,
            360,
            1.0,
            -1,
        )
        inner_shadow = cv2.GaussianBlur(inner_shadow, (11, 11), 0)

        result = result * (1 - inner_shadow * 0.03)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_eye_life(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        left_center: Tuple,
        right_center: Tuple,
        time_ms: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add subtle organic variation between eyes for lifelike quality."""

        breath_left = np.sin(time_ms * 0.0012) * 0.5 + 0.5
        breath_right = np.sin(time_ms * 0.0013 + 0.5) * 0.5 + 0.5

        if left_eye.shape == right_eye.shape:
            left_tint = 1.0 + (breath_left - 0.5) * 0.02
            right_tint = 1.0 + (breath_right - 0.5) * 0.02

            left_eye = left_eye.astype(np.float32) * left_tint
            right_eye = right_eye.astype(np.float32) * right_tint

            left_eye[:, :, 0] *= 1.0 + (breath_left - 0.5) * 0.01
            right_eye[:, :, 0] *= 1.0 + (breath_right - 0.5) * 0.01

        return np.clip(left_eye, 0, 255).astype(np.uint8), np.clip(
            right_eye, 0, 255
        ).astype(np.uint8)

    def _refine_eyelid_integration(
        self,
        eye_region: np.ndarray,
        upper_lid_mask: np.ndarray,
        lower_lid_mask: np.ndarray,
        blink_openness: float,
    ) -> np.ndarray:
        """Softer eyelid occlusion with smooth transitions."""

        combined_lid = np.maximum(upper_lid_mask, lower_lid_mask)

        combined_lid = cv2.GaussianBlur(combined_lid, (17, 17), 0)

        feather_range = 15
        kernel_size = feather_range * 2 + 1

        if blink_openness < 0.7:
            lid_edge = cv2.Laplacian(combined_lid, cv2.CV_32F)
            lid_edge = cv2.GaussianBlur(np.abs(lid_edge), (9, 9), 0)
            lid_soften = 1.0 - lid_edge * 0.3
            lid_soften = np.clip(lid_soften, 0.7, 1.0)
            eye_region = eye_region * lid_soften[:, :, np.newaxis]

        return np.clip(eye_region, 0, 255).astype(np.uint8)

    def _apply_restraint(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Subtle cinematic grading - avoid overprocessed look."""
        if intensity < 0.15:
            return frame

        result = frame.astype(np.float32)

        lift = 5 + (1 - intensity) * 10
        result = result + lift

        contrast = 1.0 + (intensity - 0.5) * 0.1
        mean = np.mean(result)
        result = mean + (result - mean) * contrast

        result = np.clip(result, 0, 255).astype(np.uint8)

        result = cv2.bilateralFilter(result, 5, 30, 30)

        return result

    def _inherit_face_lighting(
        self, eye_region: np.ndarray, face_roi: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Inherit ambient lighting from surrounding face - eyes feel same room light."""
        if face_roi is None or face_roi.size == 0:
            return eye_region

        valid_skin = (
            face_roi[mask < 0.3] if mask is not None else face_roi.reshape(-1, 3)
        )

        if len(valid_skin) < 100:
            return eye_region

        skin_mean = np.mean(valid_skin, axis=0)

        eye_mean = (
            np.mean(eye_region[eye_region > 0], axis=0)
            if np.any(eye_region > 0)
            else np.array([200, 200, 200])
        )

        brightness_ratio = np.mean(skin_mean) / (np.mean(eye_mean) + 1e-6)

        brightness_ratio = max(0.5, min(1.2, brightness_ratio))

        result = eye_region.astype(np.float32) * brightness_ratio

        color_shift = (skin_mean - eye_mean) * 0.3
        result = result + color_shift

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_eyelid_shadow_contamination(
        self,
        eye_region: np.ndarray,
        upper_lid_mask: np.ndarray,
        lower_lid_mask: np.ndarray,
    ) -> np.ndarray:
        """Eyelid shadows partially darken eyes - natural falloff."""
        if upper_lid_mask is None and lower_lid_mask is None:
            return eye_region

        lid_shadow = np.zeros_like(eye_region, dtype=np.float32)

        if upper_lid_mask is not None:
            lid_shadow += upper_lid_mask[:, :, np.newaxis] * 0.15

        if lower_lid_mask is not None:
            lid_shadow += lower_lid_mask[:, :, np.newaxis] * 0.08

        lid_shadow = cv2.GaussianBlur(lid_shadow, (15, 15), 0)

        result = eye_region.astype(np.float32) * (1 - lid_shadow * 0.4)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _soften_eye_socket_shadow(
        self, eye_region: np.ndarray, eye_center: Tuple[int, int], eye_width: int
    ) -> np.ndarray:
        """Soft, natural shadow falloff - avoid circular dark patches."""
        h, w = eye_region.shape[:2]
        cx, cy = int(eye_center[0]), int(eye_center[1])

        if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
            return eye_region

        result = eye_region.astype(np.float32)

        y_grid, x_grid = np.ogrid[:h, :w]

        dist_from_center = np.sqrt(
            ((x_grid - cx) / (w / 2)) ** 2 + ((y_grid - cy) / (h / 2)) ** 2
        )

        soft_vignette = 1.0 - (dist_from_center**2) * 0.08
        soft_vignette = np.clip(soft_vignette, 0.92, 1.0)

        result = result * soft_vignette[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_subtle_asymmetry(
        self, left_eye: np.ndarray, right_eye: np.ndarray, eye_side: str
    ) -> np.ndarray:
        """Slight differences between eyes for subtle realism."""
        noise_left = np.random.uniform(-2, 2, 3)
        noise_right = np.random.uniform(-2, 2, 3)

        if eye_side == "left":
            offset = noise_left
        else:
            offset = noise_right

        result = left_eye.astype(np.float32) + offset

        return np.clip(result, 0, 255).astype(np.uint8)

    def _blend_with_webcam_noise(
        self, eye_region: np.ndarray, original_roi: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Blend with natural webcam grain - avoid too clean look."""
        if original_roi is None or original_roi.size == 0:
            return eye_region

        edge_mask = mask < 0.5

        noise = np.random.normal(0, 3, eye_region.shape).astype(np.float32)

        result = eye_region.astype(np.float32)
        result[edge_mask] = result[edge_mask] + noise[edge_mask] * 0.15

        original_dark = original_roi.astype(np.float32) * 0.05
        result = result * 0.97 + original_dark * 0.03

        return np.clip(result, 0, 255).astype(np.uint8)


def create_ai_compositor(pack_path: Optional[str] = None) -> AIEyeCompositor:
    """Factory function for AI eye compositor."""
    return AIEyeCompositor(pack_path)
