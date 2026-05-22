"""
Professional eye compositing with perspective warping and seamless blending.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class CinematicEyeCompositor:
    """Professional eye replacement with proper 3D perspective and seamless blending."""

    def __init__(self):
        self._time = 0.0
        self._sharingan_cache = {}

    def create_sharingan_texture(self, radius: int, intensity: float) -> np.ndarray:
        """Create high-quality Sharingan texture."""
        cache_key = (radius, int(intensity * 20))
        if cache_key in self._sharingan_cache:
            return self._sharingan_cache[cache_key].copy()

        size = radius * 2 + 30
        texture = np.zeros((size, size, 4), dtype=np.uint8)
        cx, cy = size // 2, size // 2

        alpha = int(255 * min(1.0, intensity * 1.5))

        outer_r = radius
        inner_r = int(radius * 0.65)
        core_r = int(radius * 0.2)

        for y in range(size):
            for x in range(size):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist <= outer_r:
                    t = dist / outer_r
                    texture[y, x, 2] = int(180 * (1 - t * 0.5))
                    texture[y, x, 1] = int(15 * (1 - t))
                    texture[y, x, 0] = int(15 * (1 - t))
                    texture[y, x, 3] = alpha

        cv2.circle(texture, (cx, cy), outer_r, (0, 0, 0, alpha), 2, cv2.LINE_AA)
        cv2.circle(texture, (cx, cy), inner_r, (0, 0, 0, alpha), 1, cv2.LINE_AA)
        cv2.circle(texture, (cx, cy), core_r, (0, 0, 0, 0), -1, cv2.LINE_AA)

        rotation = self._time * 0.8
        for i in range(3):
            angle = rotation + (i * 2 * math.pi / 3)
            tomoye_r = max(2, int(radius * 0.12))
            tomoye_dist = int(inner_r * 0.85)

            tx = int(cx + math.cos(angle) * tomoye_dist)
            ty = int(cy + math.sin(angle) * tomoye_dist)
            cv2.circle(texture, (tx, ty), tomoye_r, (0, 0, 0, alpha), -1, cv2.LINE_AA)

            tail_angle = angle + math.pi / 2
            tail_len = tomoye_r * 3
            pts = np.array(
                [
                    [tx, ty],
                    [
                        int(tx + math.cos(tail_angle - 0.3) * tail_len),
                        int(ty + math.sin(tail_angle - 0.3) * tail_len),
                    ],
                    [
                        int(tx + math.cos(tail_angle + 0.5) * tail_len * 0.6),
                        int(ty + math.sin(tail_angle + 0.5) * tail_len * 0.6),
                    ],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(texture, [pts], (0, 0, 0, alpha))

        self._sharingan_cache[cache_key] = texture
        return texture.copy()

    def apply_perspective_transform(
        self, texture: np.ndarray, eye_angle: float, eye_tilt: float, eye_width: float
    ) -> np.ndarray:
        """Apply perspective warp based on eye orientation."""
        h, w = texture.shape[:2]

        scale_x = 1.0 - abs(math.sin(math.radians(eye_angle))) * 0.25
        new_w = int(w * scale_x)

        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        tilt_offset = int(math.sin(math.radians(eye_tilt)) * new_w * 0.15)
        dst_pts = np.float32(
            [
                [tilt_offset, 0],
                [new_w - 1 - tilt_offset, 0],
                [new_w - 1, h - 1],
                [0, h - 1],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            texture,
            matrix,
            (new_w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return cv2.resize(
            warped, (int(eye_width * 2.5), h), interpolation=cv2.INTER_LANCZOS4
        )

    def create_eyelid_mask(
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

        mask = cv2.GaussianBlur(mask, (13, 13), 0)
        return mask

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
        """Composite Sharingan with proper occlusion and blending."""

        if intensity < 0.05:
            return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        base_radius = max(10, int(iris_radius * 1.6))

        sclera_mask = self.create_eyelid_mask(upper_lid, lower_lid, (h, w))

        roi_y1 = max(0, int(eye_center[1] - eye_width))
        roi_y2 = min(h, int(eye_center[1] + eye_width))
        roi_x1 = max(0, int(eye_center[0] - eye_width))
        roi_x2 = min(w, int(eye_center[0] + eye_width))

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return result

        roi = result[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_mask = sclera_mask[roi_y1:roi_y2, roi_x1:roi_x2]

        darken = 1.0 - (0.35 * intensity)
        tinted = roi.astype(np.float32) * darken
        tinted[:, :, 0] += 20 * intensity
        result[roi_y1:roi_y2, roi_x1:roi_x2] = np.clip(tinted, 0, 255).astype(np.uint8)

        sharingan = self.create_sharingan_texture(base_radius, intensity)

        warped = self.apply_perspective_transform(sharingan, angle, tilt, eye_width)

        pw, ph = warped.shape[:2]
        cx, cy = int(iris_center[0]), int(iris_center[1])

        x1 = max(0, cx - pw // 2)
        x2 = min(w, x1 + pw)
        y1 = max(0, cy - ph // 2)
        y2 = min(h, y1 + ph)

        pw_actual = x2 - x1
        ph_actual = y2 - y1

        if pw_actual <= 0 or ph_actual <= 0:
            return result

        patch = cv2.resize(
            warped, (pw_actual, ph_actual), interpolation=cv2.INTER_LANCZOS4
        )

        local_mask = sclera_mask[y1:y2, x1:x2]
        local_mask = cv2.resize(
            local_mask, (pw_actual, ph_actual), interpolation=cv2.INTER_LINEAR
        )

        alpha = patch[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha * local_mask
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        fg = patch[:, :, :3].astype(np.float32)
        bg = result[y1:y2, x1:x2].astype(np.float32)

        alpha_3d = np.dstack([alpha, alpha, alpha])

        blended = (bg * (1 - alpha_3d) + fg * alpha_3d).astype(np.uint8)

        result[y1:y2, x1:x2] = blended

        if intensity > 0.4:
            result = self._add_ambient_glow(result, (cx, cy), eye_width, intensity)

        self._time += 0.016

        return result

    def _add_ambient_glow(
        self, frame: np.ndarray, center: Tuple, radius: float, intensity: float
    ) -> np.ndarray:
        """Add subtle ambient glow around eye."""
        cx, cy = int(center[0]), int(center[1])
        r = int(radius * 1.3)

        glow_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        cv2.circle(glow_mask, (cx, cy), r, 1.0, -1)
        glow_mask = cv2.GaussianBlur(glow_mask, (r, r), 0)

        glow_color = np.array([40, 15, 15]) * intensity * 0.4

        for c in range(3):
            ch = frame[:, :, c].astype(np.float32)
            ch = ch + glow_color[c] * glow_mask
            frame[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return frame

    def apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply cinematic color grading."""
        if intensity < 0.2:
            return frame

        result = frame.astype(np.float32)

        mean_val = np.mean(result)
        result = mean_val + (result - mean_val) * 1.12

        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        result[:, :, 0] += 10 * intensity

        return np.clip(result, 0, 255).astype(np.uint8)
