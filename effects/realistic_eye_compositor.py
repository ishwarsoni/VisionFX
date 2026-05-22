"""
Professional cinematic eye replacement with realistic compositing.
Replaces entire eye region (sclera + iris + details) for believable transformation.
"""

import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ProfessionalEyeCompositor:
    """Cinematic eye replacement using full eye textures with realistic blending."""

    def __init__(self, pack_path: Optional[str] = None):
        if pack_path is None:
            from config.assets import get_sharingan_pack_path

            pack_path = get_sharingan_pack_path()

        self.pack_path = pack_path
        self.textures = self._load_textures()
        self.current_texture = None
        self._select_random_texture()
        self._time = 0.0

        self._last_left_center = None
        self._last_right_center = None
        self._smoothing = 0.25

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

    def _extract_eye_region(
        self, frame: np.ndarray, eye_center: Tuple, eye_width: float
    ) -> Tuple:
        """Extract eye region ROI from frame."""
        h, w = frame.shape[:2]
        cx, cy = int(eye_center[0]), int(eye_center[1])
        size = int(eye_width * 2.5)

        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)

        return x1, y1, x2, y2

    def _get_full_eye_mask(
        self,
        eye_center: Tuple,
        eye_width: float,
        upper_lid: List,
        lower_lid: List,
        frame_shape: Tuple,
        intensity: float,
    ) -> np.ndarray:
        """Create comprehensive eye region mask."""
        h, w = frame_shape
        cx, cy = int(eye_center[0]), int(eye_center[1])

        mask = np.zeros((h, w), dtype=np.float32)

        size = int(eye_width * 2.2)

        upper_pts = np.array(
            [[int(p[0]), int(p[1])] for p in upper_lid if p], dtype=np.int32
        )
        lower_pts = np.array(
            [[int(p[0]), int(p[1])] for p in lower_lid if p], dtype=np.int32
        )

        if len(upper_pts) >= 3:
            for i in range(len(upper_pts)):
                p1 = upper_pts[i]
                p2 = upper_pts[(i + 1) % len(upper_pts)]
                cv2.line(mask, tuple(p1), tuple(p2), 0.6, 15)

        if len(lower_pts) >= 3:
            for i in range(len(lower_pts)):
                p1 = lower_pts[i]
                p2 = lower_pts[(i + 1) % len(lower_pts)]
                cv2.line(mask, tuple(p1), tuple(p2), 0.6, 18)

        cv2.circle(mask, (cx, cy), int(eye_width * 1.7), 1.0, -1)

        mask = cv2.GaussianBlur(mask, (19, 19), 0)

        upper_edge = np.zeros((h, w), dtype=np.float32)
        if len(upper_pts) >= 3:
            cv2.polylines(upper_edge, [upper_pts], False, 1.0, 25)
        upper_edge = cv2.GaussianBlur(upper_edge, (21, 5), 0)
        mask = mask * 0.85 + upper_edge * 0.15

        return mask

    def _prepare_full_eye_texture(
        self, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare full eye texture for replacement."""
        if self.current_texture is None:
            return None, None

        tex = self.current_texture

        h, w = target_size

        if tex.shape[2] == 4:
            bgra = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LANCZOS4)
            color = bgra[:, :, :3]
            alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        else:
            color = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LANCZOS4)
            alpha = np.ones((h, w), dtype=np.float32)

        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        return color, alpha

    def _apply_eyelid_deformation(
        self,
        color: np.ndarray,
        alpha: np.ndarray,
        eye_openness: float,
        eye_angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply realistic eyelid deformation based on eye openness."""
        h, w = color.shape[:2]

        lid_progress = 1.0 - eye_openness
        if lid_progress < 0.05:
            return color, alpha

        result_color = color.copy()
        result_alpha = alpha.copy()

        lid_height = int(h * lid_progress * 0.4)

        if lid_height > 0:
            lid_mask = np.zeros((h, w), dtype=np.float32)
            for y in range(lid_height):
                falloff = (y / lid_height) ** 2
                lid_mask[y, :] = falloff * 0.7

            lid_mask = cv2.GaussianBlur(lid_mask, (15, 15), 0)

            result_alpha = result_alpha * (1 - lid_mask)

            darken = 1.0 - lid_mask * 0.4
            result_color = (
                result_color.astype(np.float32) * darken[:, :, np.newaxis]
            ).astype(np.uint8)

        if lid_progress > 0.3:
            squeeze = 1.0 - (lid_progress - 0.3) * 0.15
            new_w = int(w * squeeze)
            offset = (w - new_w) // 2
            result_color = cv2.resize(result_color[:, offset : offset + new_w], (w, h))
            result_alpha = cv2.resize(result_alpha[:, offset : offset + new_w], (w, h))

        return result_color, result_alpha

    def _match_skin_color(
        self, eye_region: np.ndarray, surrounding: np.ndarray
    ) -> np.ndarray:
        """Match skin color for seamless blending."""
        if eye_region.size == 0 or surrounding.size == 0:
            return eye_region

        skin_mean = (
            np.mean(surrounding, axis=(0, 1))
            if surrounding.size > 0
            else np.array([128, 128, 128])
        )
        eye_mean = (
            np.mean(eye_region, axis=(0, 1))
            if eye_region.size > 0
            else np.array([128, 128, 128])
        )

        ratio = (skin_mean + 1e-6) / (eye_mean + 1e-6)

        matched = eye_region.astype(np.float32) * ratio
        matched = np.clip(matched, 0, 255).astype(np.uint8)

        return cv2.addWeighted(eye_region, 0.4, matched, 0.6, 0)

    def _apply_perspective_transform(
        self,
        color: np.ndarray,
        alpha: np.ndarray,
        eye_angle: float,
        eye_tilt: float,
        eye_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective warp to match head pose."""
        h, w = color.shape[:2]

        scale_x = 1.0 - abs(math.sin(math.radians(eye_angle))) * 0.15
        new_w = int(w * scale_x)

        tilt_offset = int(math.sin(math.radians(eye_tilt)) * new_w * 0.1)

        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32(
            [
                [tilt_offset, 0],
                [new_w - 1 - tilt_offset, 0],
                [new_w - 1, h - 1],
                [0, h - 1],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            color,
            matrix,
            (new_w, h),
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        warped_alpha = cv2.warpPerspective(
            alpha,
            matrix,
            (new_w, h),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped, warped_alpha

    def _darken_original_eye(
        self, frame: np.ndarray, mask: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Darken original eye underneath with subtle red tint."""
        if intensity < 0.2:
            return frame

        darken_factor = 1.0 - (0.35 * intensity)

        result = frame.astype(np.float32)
        result = result * darken_factor

        result[:, :, 0] += 12 * intensity
        result[:, :, 1] += 4 * intensity

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_seamless_blend(
        self, frame: np.ndarray, patch: np.ndarray, mask: np.ndarray, center: Tuple
    ) -> np.ndarray:
        """Apply cv2.seamlessClone for realistic blending."""
        h, w = frame.shape[:2]
        mask_h, mask_w = mask.shape

        if patch.shape[:2] != (mask_h, mask_w):
            patch = cv2.resize(
                patch, (mask_w, mask_h), interpolation=cv2.INTER_LANCZOS4
            )

        mask_uint8 = (mask * 255).astype(np.uint8)

        x, y = int(center[0]), int(center[1])

        try:
            result = cv2.seamlessClone(
                patch, frame, mask_uint8, (x, y), cv2.NORMAL_CLONE
            )
        except:
            result = self._alpha_blend(frame, patch, mask)

        return result

    def _alpha_blend(
        self, background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """Enhanced alpha blending with skin color matching."""
        if background.shape[:2] != alpha.shape[:2]:
            alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]))

        if foreground.shape[:2] != alpha.shape[:2]:
            fg_resized = cv2.resize(
                foreground, (background.shape[1], background.shape[0])
            )
        else:
            fg_resized = foreground

        alpha = cv2.GaussianBlur(alpha, (11, 11), 0)

        edge_mask = alpha < 0.7
        if edge_mask.any():
            skin_tone = np.median(background[edge_mask], axis=0)
            fg_edge = fg_resized[edge_mask]
            blended_tone = fg_edge.astype(np.float32) * 0.6 + skin_tone * 0.4
            fg_resized = fg_resized.astype(np.float32)
            fg_resized[edge_mask] = blended_tone
            fg_resized = np.clip(fg_resized, 0, 255).astype(np.uint8)

        alpha_3d = np.dstack([alpha, alpha, alpha])

        blended = (
            background.astype(np.float32) * (1 - alpha_3d)
            + fg_resized.astype(np.float32) * alpha_3d
        ).astype(np.uint8)

        return blended

    def _add_eye_socket_shadows(
        self, frame: np.ndarray, face_landmarks: List, intensity: float
    ) -> np.ndarray:
        """Add realistic shadows around eye sockets with soft gradient."""
        if intensity < 0.15 or not face_landmarks or len(face_landmarks) < 478:
            return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        left_eye_x = int(face_landmarks[33][0])
        right_eye_x = int(face_landmarks[263][0])
        brow_y = int(face_landmarks[10][1])

        for eye_x in [left_eye_x, right_eye_x]:
            x1 = max(0, eye_x - 60)
            x2 = min(w, eye_x + 60)
            y1 = max(0, brow_y - 40)
            y2 = min(h, brow_y + 50)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = result[y1:y2, x1:x2]

            y_grad = np.linspace(0.3, 0, y2 - y1).reshape(-1, 1)
            x_grad = np.linspace(1, 1, x2 - x1)

            gradient = np.outer(y_grad, x_grad) * intensity * 0.12

            shadow = roi.astype(np.float32)
            shadow[:, :, 0] += 18 * gradient
            shadow[:, :, 1] += 6 * gradient
            shadow[:, :, 2] += 4 * gradient

            result[y1:y2, x1:x2] = np.clip(shadow, 0, 255).astype(np.uint8)

        return result

    def _add_cinematic_bloom(
        self,
        frame: np.ndarray,
        eye_centers: List[Tuple],
        eye_width: float,
        intensity: float,
    ) -> np.ndarray:
        """Add subtle bloom with soft skin light spill."""
        if intensity < 0.25:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        for cx, cy in eye_centers:
            r = int(eye_width * 2.8)

            bloom = np.zeros((h, w), dtype=np.float32)
            cv2.circle(bloom, (int(cx), int(cy)), r, 1.0, -1)
            bloom = cv2.GaussianBlur(bloom, (r * 2, r * 2), 0)

            glow = np.array([25, 10, 10]) * intensity * 0.15

            for c in range(3):
                result[:, :, c] = np.clip(
                    result[:, :, c].astype(np.float32) + glow[c] * bloom, 0, 255
                ).astype(np.uint8)

            skin_spill = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(
                skin_spill,
                (int(cx), int(cy + eye_width * 0.5)),
                (int(eye_width * 1.5), int(eye_width * 2)),
                0,
                0,
                180,
                0.5,
                -1,
            )
            skin_spill = cv2.GaussianBlur(
                skin_spill, (int(eye_width * 1.5), int(eye_width * 1.5)), 0
            )

            spill = np.array([8, 4, 3]) * intensity * 0.08
            for c in range(3):
                result[:, :, c] = np.clip(
                    result[:, :, c].astype(np.float32) + spill[c] * skin_spill, 0, 255
                ).astype(np.uint8)

        return result

    def _add_reflections(
        self,
        frame: np.ndarray,
        eye_centers: List[Tuple],
        eye_width: float,
        intensity: float,
    ) -> np.ndarray:
        """Add subtle corneal reflections with slight asymmetry."""
        if intensity < 0.4:
            return frame

        result = frame.copy()

        for i, (cx, cy) in enumerate(eye_centers):
            r = int(eye_width * 0.2)

            offset_x = -r * 0.4 if i == 0 else -r * 0.35
            offset_y = -r * 0.35 if i == 0 else -r * 0.4

            reflect_mask = np.zeros(frame.shape[:2], dtype=np.float32)
            cv2.circle(
                reflect_mask, (int(cx + offset_x), int(cy + offset_y)), r, 1.0, -1
            )
            reflect_mask = cv2.GaussianBlur(reflect_mask, (7, 7), 0)

            reflect = result.astype(np.float32)
            reflect += 15 * reflect_mask[:, :, np.newaxis]

            result = np.clip(reflect, 0, 255).astype(np.uint8)

        return result

    def _apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply subtle cinematic color grading."""
        if intensity < 0.1:
            return frame

        result = frame.astype(np.float32)

        result[:, :, 0] += 8 * intensity
        result[:, :, 1] += 3 * intensity

        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        vignette = (
            1
            - (
                np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
                / np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
            )
            * intensity
            * 0.15
        )
        vignette = np.clip(vignette, 0, 1)
        result = result * vignette[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    def composite_eyes(
        self,
        frame: np.ndarray,
        left_eye_data: dict,
        right_eye_data: dict,
        face_landmarks: List,
        intensity: float,
    ) -> np.ndarray:
        """Composite full eye replacements with cinematic realism."""

        if intensity < 0.05:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        eye_centers = []

        for eye_data in [left_eye_data, right_eye_data]:
            if not eye_data.get("detected"):
                continue

            eye_center = eye_data["center"]
            eye_width = eye_data["width"]
            upper_lid = eye_data.get("upper_lid", [])
            lower_lid = eye_data.get("lower_lid", [])
            angle = eye_data.get("angle", 0)
            tilt = eye_data.get("tilt", 0)
            eye_openness = eye_data.get("openness", 1.0)

            eye_centers.append(eye_center)

            x1, y1, x2, y2 = self._extract_eye_region(result, eye_center, eye_width)

            if x2 <= x1 or y2 <= y1:
                continue

            roi_w = x2 - x1
            roi_h = y2 - y1

            texture, alpha = self._prepare_full_eye_texture((roi_h, roi_w))
            if texture is None:
                continue

            warped, warped_alpha = self._apply_perspective_transform(
                texture, alpha, angle, tilt, eye_width
            )

            warped, warped_alpha = self._apply_eyelid_deformation(
                warped, warped_alpha, eye_openness, angle
            )

            if warped.shape[:2] != (roi_h, roi_w):
                warped = cv2.resize(warped, (roi_w, roi_h))
                warped_alpha = cv2.resize(warped_alpha, (roi_w, roi_h))

            roi = result[y1:y2, x1:x2].copy()

            darken = 1.0 - (0.45 * intensity)
            tinted_roi = roi.astype(np.float32) * darken
            tinted_roi[:, :, 0] += 20 * intensity
            tinted_roi[:, :, 1] += 8 * intensity
            result[y1:y2, x1:x2] = np.clip(tinted_roi, 0, 255).astype(np.uint8)

            mask = self._get_full_eye_mask(
                eye_center, eye_width, upper_lid, lower_lid, (h, w), intensity
            )
            local_mask = mask[y1:y2, x1:x2]
            local_mask = cv2.resize(
                local_mask, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR
            )

            final_alpha = warped_alpha * local_mask

            fg = warped.astype(np.float32)
            bg = result[y1:y2, x1:x2].astype(np.float32)

            fg_matched = self._match_skin_color(
                fg.astype(np.uint8), bg.astype(np.uint8)
            ).astype(np.float32)

            alpha_3d = np.dstack([final_alpha, final_alpha, final_alpha])

            blended = (bg * (1 - alpha_3d) + fg_matched * alpha_3d).astype(np.uint8)
            result[y1:y2, x1:x2] = blended

        if intensity > 0.3:
            result = self._add_eye_socket_shadows(result, face_landmarks, intensity)

        if intensity > 0.4:
            result = self._add_cinematic_bloom(
                result,
                eye_centers,
                (left_eye_data.get("width", 50) + right_eye_data.get("width", 50)) / 2,
                intensity,
            )

        if intensity > 0.5:
            result = self._add_reflections(
                result,
                eye_centers,
                (left_eye_data.get("width", 50) + right_eye_data.get("width", 50)) / 2,
                intensity,
            )

        if intensity > 0.15:
            result = self._apply_cinematic_grading(result, intensity)

        self._time += 0.016

        return result


class WebcamQualityEnhancer:
    """Subtle cinematic webcam enhancement - avoid plastic/AI filter look."""

    def __init__(self):
        self._denoise_strength = 1
        self._clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle quality improvements - preserve natural skin texture."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            return frame

        result = frame.astype(np.float32)

        bright = np.mean(result)
        if bright < 85:
            gain = 1.0 + (90 - bright) / 500
            result = result * gain
            result = np.clip(result, 0, 255)

        luma = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2YUV)[:, :, 0]
        luma = self._clahe.apply(luma)
        yuv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = luma
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.float32)

        lift = 2
        result = result + lift

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


def create_professional_compositor(
    pack_path: Optional[str] = None,
) -> ProfessionalEyeCompositor:
    """Factory function."""
    return ProfessionalEyeCompositor(pack_path)


def create_webcam_enhancer() -> WebcamQualityEnhancer:
    """Factory function."""
    return WebcamQualityEnhancer()
