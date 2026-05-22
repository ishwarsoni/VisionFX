"""
Professional anime eye replacement using Sharingan Pack textures.
Uses proper perspective warping, eyelid occlusion, and seamless blending.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from effects.eye_texture_loader import (
    EyeTexture,
    SharinganTextureLoader,
    create_texture_loader,
)


class TextureBasedEyeCompositor:
    """Professional eye replacement using actual anime eye textures."""

    def __init__(self, pack_path: Optional[str] = None):
        if pack_path is None:
            from config.assets import get_sharingan_pack_path

            pack_path = get_sharingan_pack_path()

        self.texture_loader = create_texture_loader(pack_path)
        self.current_texture: Optional[EyeTexture] = None
        self._texture_cache = {}
        self._time = 0.0

    def select_eye_style(self, style: str = "random") -> None:
        """Select eye style from available textures."""
        if style == "random":
            self.current_texture = self.texture_loader.get_random_texture()
        elif style == "next":
            self.current_texture = self.texture_loader.get_next_texture()
        else:
            self.current_texture = self.texture_loader.get_texture_by_name(style)

    def _prepare_texture(
        self, texture: np.ndarray, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare texture for compositing."""
        h, w = texture.shape[:2]

        if texture.shape[2] == 4:
            bgra = texture
        else:
            bgra = cv2.cvtColor(texture, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = 255

        resized = cv2.resize(bgra, target_size, interpolation=cv2.INTER_LANCZOS4)

        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        return resized[:, :, :3], alpha

    def apply_perspective_warp(
        self,
        texture: np.ndarray,
        alpha: np.ndarray,
        eye_angle: float,
        eye_tilt: float,
        eye_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective warp based on eye orientation."""
        h, w = texture.shape[:2]

        scale_x = 1.0 - abs(math.sin(math.radians(eye_angle))) * 0.2
        new_w = int(w * scale_x)

        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        tilt_offset = int(math.sin(math.radians(eye_tilt)) * new_w * 0.12)
        dst_pts = np.float32(
            [
                [tilt_offset, 0],
                [new_w - 1 - tilt_offset, 0],
                [new_w - 1, h - 1],
                [0, h - 1],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_color = cv2.warpPerspective(
            texture,
            matrix,
            (new_w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        warped_alpha = cv2.warpPerspective(
            alpha,
            matrix,
            (new_w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped_color, warped_alpha

    def _create_eyelid_mask(
        self, upper_lid, lower_lid, frame_shape: Tuple
    ) -> np.ndarray:
        """Create soft eyelid occlusion mask."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.float32)

        upper_pts = np.array(
            [[int(p[0]), int(p[1])] for p in upper_lid], dtype=np.int32
        )
        lower_pts = np.array(
            [[int(p[0]), int(p[1])] for p in lower_lid], dtype=np.int32
        )

        if len(upper_pts) >= 3 and len(lower_pts) >= 3:
            contour = np.vstack([upper_pts, lower_pts[::-1]])
            cv2.fillPoly(mask, [contour], 1.0)

        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        return mask

    def _darken_sclera(
        self,
        frame: np.ndarray,
        upper_lid,
        lower_lid,
        eye_center,
        eye_width,
        intensity: float,
    ) -> np.ndarray:
        """Darken sclera for supernatural effect."""
        h, w = frame.shape[:2]
        result = frame.copy()

        mask = self._create_eyelid_mask(upper_lid, lower_lid, (h, w))

        y1 = max(0, int(eye_center[1] - eye_width))
        y2 = min(h, int(eye_center[1] + eye_width))
        x1 = max(0, int(eye_center[0] - eye_width))
        x2 = min(w, int(eye_center[0] + eye_width))

        if y2 <= y1 or x2 <= x1:
            return result

        roi = result[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]

        if roi.size == 0:
            return result

        darken = 1.0 - (0.4 * intensity)
        tinted = roi.astype(np.float32) * darken
        tinted[:, :, 0] += 25 * intensity
        tinted[:, :, 1] += 10 * intensity

        result[y1:y2, x1:x2] = np.where(
            roi_mask[:, :, np.newaxis] > 0.1,
            np.clip(tinted, 0, 255).astype(np.uint8),
            roi,
        )

        return result

    def composite_eye(
        self,
        frame: np.ndarray,
        iris_center: Tuple,
        iris_radius: float,
        eye_center: Tuple,
        eye_width: float,
        upper_lid,
        lower_lid,
        angle: float,
        tilt: float,
        intensity: float,
    ) -> np.ndarray:
        """Composite anime eye texture with proper blending."""

        if intensity < 0.05:
            return frame

        if self.current_texture is None:
            self.select_eye_style("random")

        if self.current_texture is None:
            return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        eye_size = int(eye_width * 2.2)
        eye_size = max(30, min(eye_size, 300))

        result = self._darken_sclera(
            result, upper_lid, lower_lid, eye_center, eye_width, intensity
        )

        tex, alpha = self._prepare_texture(
            self.current_texture.texture, (eye_size, eye_size)
        )

        warped_color, warped_alpha = self.apply_perspective_warp(
            tex, alpha, angle, tilt, eye_width
        )

        pw, ph = warped_color.shape[:2]

        cx, cy = int(iris_center[0]), int(iris_center[1])

        x1 = max(0, cx - pw // 2)
        x2 = min(w, x1 + pw)
        y1 = max(0, cy - ph // 2)
        y2 = min(h, y1 + ph)

        pw_actual = x2 - x1
        ph_actual = y2 - y1

        if pw_actual <= 0 or ph_actual <= 0:
            return result

        patch_color = cv2.resize(
            warped_color, (pw_actual, ph_actual), interpolation=cv2.INTER_LANCZOS4
        )
        patch_alpha = cv2.resize(
            warped_alpha, (pw_actual, ph_actual), interpolation=cv2.INTER_LINEAR
        )

        eyelid_mask = self._create_eyelid_mask(upper_lid, lower_lid, (h, w))
        local_eyelid = eyelid_mask[y1:y2, x1:x2]
        local_eyelid = cv2.resize(
            local_eyelid, (pw_actual, ph_actual), interpolation=cv2.INTER_LINEAR
        )

        final_alpha = patch_alpha * local_eyelid
        final_alpha = cv2.GaussianBlur(final_alpha, (3, 3), 0)

        fg = patch_color.astype(np.float32)
        bg = result[y1:y2, x1:x2].astype(np.float32)

        alpha_3d = np.dstack([final_alpha, final_alpha, final_alpha])

        blended = (bg * (1 - alpha_3d) + fg * alpha_3d).astype(np.uint8)
        result[y1:y2, x1:x2] = blended

        if intensity > 0.5:
            result = self._add_ambient_glow(result, (cx, cy), eye_width, intensity)

        self._time += 0.016

        return result

    def _add_ambient_glow(
        self, frame: np.ndarray, center: Tuple, radius: float, intensity: float
    ) -> np.ndarray:
        """Add cinematic ambient glow."""
        cx, cy = int(center[0]), int(center[1])
        r = int(radius * 1.4)

        glow_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        cv2.circle(glow_mask, (cx, cy), r, 1.0, -1)
        glow_mask = cv2.GaussianBlur(glow_mask, (r, r), 0)

        glow_color = np.array([50, 20, 20]) * intensity * 0.35

        for c in range(3):
            ch = frame[:, :, c].astype(np.float32)
            ch = ch + glow_color[c] * glow_mask
            frame[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return frame

    def apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply cinematic color grading."""
        if intensity < 0.15:
            return frame

        result = frame.astype(np.float32)

        mean_val = np.mean(result)
        result = mean_val + (result - mean_val) * 1.1

        result[:, :, 0] += 12 * intensity
        result[:, :, 1] += 5 * intensity

        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist / max_dist) * intensity * 0.25
        vignette = np.clip(vignette, 0, 1)
        result = result * vignette[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)


def create_texture_compositor(
    pack_path: Optional[str] = None,
) -> TextureBasedEyeCompositor:
    """Factory function to create texture-based compositor."""
    return TextureBasedEyeCompositor(pack_path)
