"""
Cinematic eye VFX renderer — professional real-time Sharingan pipeline.

Implements a production-inspired pipeline that:
- extracts full eye region mask
- detects iris/pupil ellipse
- generates a procedural Sharingan texture (fallback)
- warps and blends the texture to match pupil geometry
- applies eyelid occlusion and feathering
- darkens sclera and adds ambient spill + bloom
- uses temporal EMA smoothing for stable tracking

This module is written to integrate with the existing TrackingFrameData structures
and uses the eye landmarks and iris data already computed by the tracking stack.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from tracking.tracking_data import IrisData, TrackingFrameData
from utils.logger import Logger


def _feather_mask(mask: np.ndarray, ksize: int = 21) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)


def _ellipse_from_contour(
    contour: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    try:
        if contour is None or len(contour) < 5:
            return None
        ellipse = cv2.fitEllipse(contour)
        return ellipse  # ((cx, cy), (major, minor), angle)
    except Exception:
        return None


class EyeVFXRenderer:
    """Cinematic Sharingan renderer for one frame.

    Usage: create an instance and call `process(frame, tracking_data)`.
    Maintains per-eye EMA state for smoothing.
    """

    def __init__(
        self, logger: Optional[Logger] = None, texture: Optional[np.ndarray] = None
    ):
        self.logger = logger or Logger(name="EyeVFXRenderer")
        self.logger.info("EyeVFXRenderer initialized")

        # State for temporal smoothing: left/right
        self.state = {
            "left": {"center": None, "axes": None, "angle": None},
            "right": {"center": None, "axes": None, "angle": None},
        }

        # Sharingan texture (BGR). If None, generate procedurally on demand.
        self.base_texture = texture

        # Parameters
        self.smooth_alpha = 0.35  # EMA alpha
        self.feather_px = 15
        self.bloom_strength = 0.6

    def process(
        self, frame: np.ndarray, tracking_data: TrackingFrameData
    ) -> np.ndarray:
        if not tracking_data.is_tracking() or not tracking_data.face.detected:
            return frame

        out = frame.copy()

        # Process each eye independently
        for side in ("left", "right"):
            eye = getattr(tracking_data, f"{side}_eye")
            if not eye.landmarks or eye.iris.confidence < 0.1:
                continue

            try:
                out = self._process_eye(out, eye.landmarks, eye.iris, side)
            except Exception as e:
                self.logger.debug(f"Eye VFX error ({side}): {e}")

        return out

    def _process_eye(
        self, frame: np.ndarray, eye_landmarks: list, iris: IrisData, side: str
    ) -> np.ndarray:
        # Compute eye bounding box
        xs = [p[0] for p in eye_landmarks]
        ys = [p[1] for p in eye_landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Expand ROI slightly for grades and spill
        pad_x = int((x_max - x_min) * 0.7) + 8
        pad_y = int((y_max - y_min) * 0.7) + 8
        rx0 = max(0, x_min - pad_x)
        ry0 = max(0, y_min - pad_y)
        rx1 = min(frame.shape[1], x_max + pad_x)
        ry1 = min(frame.shape[0], y_max + pad_y)

        roi = frame[ry0:ry1, rx0:rx1]

        # Convert landmarks into ROI-local coordinates
        lm = [(int(x - rx0), int(y - ry0)) for (x, y) in eye_landmarks]

        # Eye mask
        eye_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.array(lm, dtype=np.int32))
        cv2.fillConvexPoly(eye_mask, hull, 255)

        # Detect pupil contour inside ROI using simple thresholding on ROI
        pupil_contour = self._detect_pupil_contour(roi, lm)
        ellipse = _ellipse_from_contour(pupil_contour)

        # Fallback: approximate ellipse from iris normalized center and ratio
        if ellipse is None:
            # Compute normalized center in ROI
            eye_w = x_max - x_min
            eye_h = y_max - y_min
            cx = int((iris.center[0] * eye_w) + (min(xs) - rx0))
            cy = int((iris.center[1] * eye_h) + (min(ys) - ry0))
            radius = int(max(eye_w, eye_h) * iris.radius)
            ellipse = ((cx, cy), (radius * 2, radius * 2), 0.0)

        # Smooth ellipse parameters temporally
        (ecx, ecy), (ewa, ehb), eangle = ellipse
        state = self.state[side]
        if state["center"] is None:
            state["center"] = np.array([ecx, ecy], dtype=np.float32)
            state["axes"] = np.array([ewa, ehb], dtype=np.float32)
            state["angle"] = float(eangle)
        else:
            state["center"] = (1 - self.smooth_alpha) * state[
                "center"
            ] + self.smooth_alpha * np.array([ecx, ecy], dtype=np.float32)
            state["axes"] = (1 - self.smooth_alpha) * state[
                "axes"
            ] + self.smooth_alpha * np.array([ewa, ehb], dtype=np.float32)
            state["angle"] = (1 - self.smooth_alpha) * state[
                "angle"
            ] + self.smooth_alpha * float(eangle)

        # Build sharingan layer sized to axes
        axes = tuple([max(2, int(x)) for x in state["axes"]])
        center = tuple([int(x) for x in state["center"]])
        angle = float(state["angle"])

        sharingan = self._build_sharingan_texture(max(axes[0], axes[1]))

        # Rotate and scale texture to match ellipse axes and angle
        scaled = cv2.resize(sharingan, (axes[0], axes[1]), interpolation=cv2.INTER_AREA)
        M = cv2.getRotationMatrix2D((axes[0] // 2, axes[1] // 2), angle, 1.0)
        warped_tex = cv2.warpAffine(
            scaled,
            M,
            (axes[0], axes[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Create ellipse mask
        tex_mask = np.zeros((axes[1], axes[0]), dtype=np.uint8)
        cv2.ellipse(
            tex_mask,
            (axes[0] // 2, axes[1] // 2),
            (axes[0] // 2, axes[1] // 2),
            0,
            0,
            360,
            255,
            -1,
        )
        tex_mask = _feather_mask(tex_mask, ksize=self.feather_px)
        tex_mask = (tex_mask / tex_mask.max()).astype(np.float32)

        # Place texture into ROI
        layer = np.zeros_like(roi, dtype=np.uint8)
        x0 = int(center[0] - axes[0] // 2)
        y0 = int(center[1] - axes[1] // 2)
        x1 = x0 + axes[0]
        y1 = y0 + axes[1]

        # Bounds check and clip
        sx0, sy0, dx0, dy0 = 0, 0, x0, y0
        sx1, sy1 = axes[0], axes[1]
        if x0 < 0:
            sx0 = -x0
            dx0 = 0
        if y0 < 0:
            sy0 = -y0
            dy0 = 0
        if x1 > roi.shape[1]:
            sx1 = axes[0] - (x1 - roi.shape[1])
        if y1 > roi.shape[0]:
            sy1 = axes[1] - (y1 - roi.shape[0])

        if sx1 <= sx0 or sy1 <= sy0:
            return frame

        layer[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = warped_tex[
            sy0:sy1, sx0:sx1
        ]
        mask_patch = tex_mask[sy0:sy1, sx0:sx1]

        # Eyelid occlusion: compute upper/lower eyelid masks in ROI
        eyelid_mask = self._compute_eyelid_occlusion_mask(roi.shape[:2], lm, center)

        # Final mask: ellipse mask * eye_mask * (1 - eyelid_mask)
        final_mask = np.zeros(roi.shape[:2], dtype=np.float32)
        # paste mask_patch into final_mask at dx0,dy0
        final_mask[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = mask_patch
        final_mask = final_mask * (eye_mask.astype(np.float32) / 255.0)
        final_mask = final_mask * (1.0 - eyelid_mask)

        # Add bloom: extract bright parts of layer and blur
        bright = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        bright = np.clip((bright - 0.6) * 3.0, 0.0, 1.0)
        bloom = cv2.GaussianBlur(
            (layer.astype(np.float32) / 255.0) * bright[..., None],
            (0, 0),
            sigmaX=12,
            sigmaY=12,
        )
        bloom = np.clip(bloom * self.bloom_strength, 0.0, 1.0)

        # Composite: seamlessClone expects 8-bit masks
        mask_u8 = (final_mask * 255).astype(np.uint8)

        # Prepare patch for seamlessClone
        patch = (layer.astype(np.float32) * final_mask[..., None]).astype(np.uint8)

        # Use Poisson cloning for natural blend when possible
        try:
            center_point = (int(center[0] + rx0), int(center[1] + ry0))
            clone = cv2.seamlessClone(
                patch, frame, mask_u8, center_point, cv2.NORMAL_CLONE
            )
            frame = clone
        except Exception:
            # Fallback to alpha blend into ROI
            alpha = final_mask[..., None]
            roi_float = roi.astype(np.float32) / 255.0
            layer_float = layer.astype(np.float32) / 255.0
            composited = layer_float * alpha + roi_float * (1 - alpha)
            # add bloom
            composited = np.clip(composited + bloom, 0.0, 1.0)
            frame[ry0:ry1, rx0:rx1] = (composited * 255).astype(np.uint8)

        # Sclera darkening and grading
        frame = self._grade_sclera(frame, rx0, ry0, rx1, ry1, eye_mask, final_mask)

        # Ambient red spill around eye
        frame = self._apply_ambient_spill(frame, rx0, ry0, rx1, ry1, center, final_mask)

        return frame

    def _detect_pupil_contour(self, roi: np.ndarray, lm: list):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # threshold low values (dark pupil)
            _, thresh = cv2.threshold(
                enhanced, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # mask by eye hull
            hull = cv2.convexHull(np.array(lm, dtype=np.int32))
            mask = np.zeros_like(thresh)
            cv2.fillConvexPoly(mask, hull, 255)
            thresh = cv2.bitwise_and(thresh, mask)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            # pick largest
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 10:
                return None
            return c
        except Exception:
            return None

    def _compute_eyelid_occlusion_mask(
        self, size: Tuple[int, int], lm: list, pupil_center: Tuple[int, int]
    ) -> np.ndarray:
        h, w = size
        mask = np.zeros((h, w), dtype=np.float32)

        # Split landmarks into upper and lower by y relative to pupil
        upper = [pt for pt in lm if pt[1] < pupil_center[1]]
        lower = [pt for pt in lm if pt[1] >= pupil_center[1]]

        if len(upper) >= 3:
            uhull = cv2.convexHull(np.array(upper, dtype=np.int32))
            up = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(up, uhull, 255)
            upf = _feather_mask(up, ksize=self.feather_px)
            mask = np.maximum(mask, upf / 255.0)

        if len(lower) >= 3:
            lhull = cv2.convexHull(np.array(lower, dtype=np.int32))
            lp = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(lp, lhull, 255)
            lpf = _feather_mask(lp, ksize=self.feather_px)
            mask = np.maximum(mask, lpf / 255.0)

        # Clamp
        mask = np.clip(mask, 0.0, 1.0)
        return mask

    def _build_sharingan_texture(self, diameter: int) -> np.ndarray:
        # If user provided base texture, scale and return
        if self.base_texture is not None:
            tex = cv2.resize(
                self.base_texture, (diameter, diameter), interpolation=cv2.INTER_AREA
            )
            return tex

        # Procedural generation: concentric rings + subtle pattern
        d = max(4, int(diameter))
        tex = np.zeros((d, d, 3), dtype=np.uint8)
        cx, cy = d // 2, d // 2
        # radial gradient
        y, x = np.ogrid[:d, :d]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_norm = (r / (d / 2)).clip(0, 1)
        base = (1.0 - r_norm) ** 2
        # red base
        tex[..., 2] = (base * 200).astype(np.uint8)
        # add dark pupil center
        cv2.circle(tex, (cx, cy), int(d * 0.18), (10, 10, 10), -1)
        # add subtle rings
        for i in range(3):
            cv2.circle(tex, (cx, cy), int(d * (0.18 + 0.06 * (i + 1))), (40, 0, 0), 2)
        # add stylized tomoe shapes (simple triangles)
        for t in range(3):
            ang = np.deg2rad(90 + t * 120)
            r1 = int(d * 0.32)
            p1 = (int(cx + np.cos(ang) * r1), int(cy + np.sin(ang) * r1))
            p2 = (
                int(cx + np.cos(ang + 0.25) * r1 * 0.6),
                int(cy + np.sin(ang + 0.25) * r1 * 0.6),
            )
            cv2.ellipse(
                tex,
                p1,
                (int(d * 0.06), int(d * 0.12)),
                np.rad2deg(ang),
                0,
                180,
                (20, 20, 20),
                -1,
            )

        # subtle noise overlay
        noise = (np.random.randn(d, d) * 8).astype(np.int16)
        for c in range(3):
            ch = tex[..., c].astype(np.int16) + noise
            tex[..., c] = np.clip(ch, 0, 255).astype(np.uint8)

        return tex

    def _grade_sclera(self, frame, rx0, ry0, rx1, ry1, eye_mask, final_mask):
        roi = frame[ry0:ry1, rx0:rx1]
        # sclera mask = eye_mask - final_mask
        final_mask_u = (final_mask > 0.01).astype(np.uint8)
        sclera = (eye_mask.astype(np.uint8) - (final_mask_u * 255)).clip(0, 255)
        sclera_f = _feather_mask(sclera, ksize=self.feather_px) / 255.0

        # darken sclera and add subtle red tint
        roi_f = roi.astype(np.float32) / 255.0
        dark = roi_f * (1.0 - 0.45 * sclera_f[..., None])
        tint = np.zeros_like(roi_f)
        tint[..., 2] = 0.12 * sclera_f
        out = np.clip(dark + tint, 0.0, 1.0)
        frame[ry0:ry1, rx0:rx1] = (out * 255).astype(np.uint8)
        return frame

    def _apply_ambient_spill(self, frame, rx0, ry0, rx1, ry1, center, final_mask):
        # Create a soft red spill around the eye region
        cx = int(center[0] + rx0)
        cy = int(center[1] + ry0)
        overlay = frame.copy().astype(np.float32) / 255.0
        h, w = frame.shape[:2]
        spill = np.zeros((h, w), dtype=np.float32)
        rr = int(max(rx1 - rx0, ry1 - ry0) * 0.9)
        cv2.circle(spill, (cx, cy), rr, 1.0, -1)
        spill = cv2.GaussianBlur(spill, (0, 0), sigmaX=rr * 0.6, sigmaY=rr * 0.6)
        # colorize and composite
        color = np.zeros_like(overlay)
        color[..., 2] = 0.22  # red
        composite = overlay + color * spill[..., None] * 0.25
        frame = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
        return frame
