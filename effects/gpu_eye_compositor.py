"""
GPU-Accelerated Eye Compositor
Uses ModernGL for fast shader-based eye rendering and compositing.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import moderngl
    import moderngl_window as mglw

    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False


class GPUEyeCompositor:
    """
    GPU-accelerated eye compositor using ModernGL shaders.
    Provides fast eye warping, reflections, and cinematic compositing.
    """

    VERTEX_SHADER = """
    #version 330
    in vec2 in_texcoord;
    in vec2 in_position;
    out vec2 v_texcoord;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    """

    EYE_FRAGMENT_SHADER = """
    #version 330
    uniform sampler2D eye_texture;
    uniform sampler2D alpha_texture;
    uniform sampler2D mask_texture;
    uniform vec2 iris_offset;
    uniform float blink_openness;
    uniform float time;
    uniform float intensity;
    uniform float eye_width;
    uniform vec3 scene_lighting;
    in vec2 v_texcoord;
    out vec4 f_color;
    
    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }
    
    void main() {
        vec2 uv = v_texcoord;
        
        float blink_compress = 1.0 - (1.0 - blink_openness) * 0.2;
        uv.y = (uv.y - 0.5) / blink_compress + 0.5;
        
        uv.x += iris_offset.x * 0.1;
        uv.y += iris_offset.y * 0.06;
        
        vec2 center = vec2(0.5, 0.5);
        float dist_from_center = distance(uv, center);
        
        float bulge = 1.0 + 0.06 * (1.0 - dist_from_center * 1.6);
        uv = center + (uv - center) / bulge;
        
        vec4 eye_color = texture(eye_texture, uv);
        float alpha = texture(alpha_texture, uv).r;
        float mask = texture(mask_texture, uv).r;
        
        float sclera_shade = 1.0 - dist_from_center * 0.1;
        eye_color.rgb = eye_color.rgb * sclera_shade;
        
        float inner_shadow = smoothstep(0.0, 0.4, dist_from_center) * 0.08;
        eye_color.rgb = eye_color.rgb * (1.0 - inner_shadow);
        
        float lid_occlusion = 0.0;
        if (uv.y < 0.32 * blink_openness + 0.06) {
            lid_occlusion = 1.0 - smoothstep(0.0, 0.15, uv.y);
        }
        
        float final_alpha = alpha * mask * (1.0 - lid_occlusion * intensity);
        
        float breathe = sin(time * 1.2) * 0.5 + 0.5;
        
        float slow_drift = sin(time * 0.4) * 0.003;
        
        vec2 spec_offset = vec2(
            sin(time * 1.0 + 0.3) * 0.006 + slow_drift,
            cos(time * 0.7 + 0.5) * 0.005
        );
        
        float spec = 0.0;
        vec2 spec_pos = vec2(0.35 + sin(time * 0.2) * 0.02, 0.32) + spec_offset;
        float spec_dist = distance(uv, spec_pos);
        if (spec_dist < 0.12) {
            spec = pow(1.0 - spec_dist / 0.12, 2.5) * 0.1 * (0.35 + 0.65 * breathe);
        }
        
        vec2 spec2_pos = vec2(0.62 + cos(time * 0.15) * 0.015, 0.45) + spec_offset * 0.7;
        float spec2_dist = distance(uv, spec2_pos);
        if (spec2_dist < 0.07) {
            spec += pow(1.0 - spec2_dist / 0.07, 2.5) * 0.05;
        }
        
        vec2 spec3_pos = vec2(0.25, 0.6 + sin(time * 0.3) * 0.02);
        float spec3_dist = distance(uv, spec3_pos);
        if (spec3_dist < 0.05) {
            spec += pow(1.0 - spec3_dist / 0.05, 2.0) * 0.03;
        }
        
        vec3 room_tint = vec3(1.03, 1.0, 0.97);
        eye_color.rgb = eye_color.rgb * room_tint;
        
        eye_color.rgb += vec3(spec) * scene_lighting;
        
        float grain = (random(uv + time * 0.008) - 0.5) * 0.025;
        eye_color.rgb += grain;
        
        float edge = smoothstep(0.0, 0.1, final_alpha);
        float soft_edge = smoothstep(0.0, 0.06, final_alpha);
        
        f_color = vec4(eye_color.rgb, final_alpha * edge * 0.92 + soft_edge * final_alpha * 0.08);
    }
    """

    BLUR_FRAGMENT_SHADER = """
    #version 330
    uniform sampler2D texture0;
    uniform vec2 resolution;
    uniform float strength;
    in vec2 v_texcoord;
    out vec4 f_color;
    
    void main() {
        vec2 texel = 1.0 / resolution;
        vec4 color = vec4(0.0);
        float total = 0.0;
        
        for (float x = -4.0; x <= 4.0; x += 1.0) {
            for (float y = -4.0; y <= 4.0; y += 1.0) {
                float weight = exp(-(x*x + y*y) / (2.0 * strength * strength));
                color += texture(texture0, v_texcoord + vec2(x, y) * texel) * weight;
                total += weight;
            }
        }
        
        f_color = color / total;
    }
    """

    def __init__(self, pack_path: Optional[str] = None):
        if pack_path is None:
            from config.assets import get_sharingan_pack_path

            pack_path = get_sharingan_pack_path()

        self.pack_path = pack_path
        self.ctx = None
        self._init_gpu()

        self.textures = self._load_textures()
        self._select_random_texture()

        self._time = 0.0

    def _init_gpu(self):
        """Initialize ModernGL context."""
        if not MODERNGL_AVAILABLE:
            return

        try:
            self.ctx = moderngl.create_standalone_context()

            self.eye_program = self.ctx.program(
                vertex_shader=self.VERTEX_SHADER,
                fragment_shader=self.EYE_FRAGMENT_SHADER,
            )

            self.blur_program = self.ctx.program(
                vertex_shader=self.VERTEX_SHADER,
                fragment_shader=self.BLUR_FRAGMENT_SHADER,
            )

            self.quad_buffer = self.ctx.buffer(
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],
                    dtype="f4",
                )
            )

            self._gpu_available = True

        except Exception as e:
            self._gpu_available = False

    def _load_textures(self) -> List[np.ndarray]:
        """Load realistic eye textures."""
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

    def gpu_blur(self, texture_data: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply GPU-based Gaussian blur."""
        if not self._gpu_available or texture_data is None:
            return cv2.GaussianBlur(texture_data, (15, 15), strength)

        try:
            h, w = texture_data.shape[:2]

            tex = self.ctx.texture((w, h), 3, texture_data.tobytes())
            fbo = self.ctx.framebuffer(color_attachments=[tex])

            self.blur_program["resolution"].value = (w, h)
            self.blur_program["strength"].value = strength

            fbo.use()
            tex.use(0)

            vao = self.ctx.vertex_array(
                self.blur_program,
                [(self.quad_buffer, "2f 2f", "in_texcoord", "in_position")],
            )
            vao.render(moderngl.TRIANGLE_STRIP)

            data = fbo.read(components=3)
            return np.frombuffer(data, dtype="u1").reshape(h, w, 3)

        except:
            return cv2.GaussianBlur(texture_data, (15, 15), strength)

    def composite_eyes(
        self,
        frame: np.ndarray,
        left_eye_data: dict,
        right_eye_data: dict,
        landmarks: List,
        intensity: float,
    ) -> np.ndarray:
        """Composite eyes with GPU acceleration fallback to CPU."""

        if intensity < 0.05:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        eye_centers = []

        for eye_data in [left_eye_data, right_eye_data]:
            if not eye_data.get("detected"):
                continue

            center = eye_data.get("iris_center", eye_data.get("center"))
            eye_width = eye_data.get("width", 60)

            if not center:
                continue

            eye_centers.append(center)

            scale_factor = 1.35
            scaled_width = eye_width * scale_factor

            x1 = max(0, int(center[0] - scaled_width))
            x2 = min(w, int(center[0] + scaled_width))
            y1 = max(0, int(center[1] - scaled_width * 0.9))
            y2 = min(h, int(center[1] + scaled_width * 0.9))

            if x2 <= x1 or y2 <= y1:
                continue

            roi_w = x2 - x1
            roi_h = y2 - y1

            if self.current_texture is None:
                continue

            tex = cv2.resize(self.current_texture, (roi_w, roi_h))

            if tex.shape[2] == 4:
                color = tex[:, :, :3]
                alpha = tex[:, :, 3].astype(np.float32) / 255.0
            else:
                color = tex
                alpha = np.ones((roi_h, roi_w), np.float32)

            alpha = cv2.GaussianBlur(alpha, (9, 9), 0)

            eye_region = np.zeros((roi_h, roi_w), dtype=np.float32)
            cx, cy = roi_w // 2, roi_h // 2

            y_grid, x_grid = np.ogrid[:roi_h, :roi_w]
            dx = (x_grid - cx) / (roi_w / 2)
            dy = (y_grid - cy) / (roi_h / 2)
            dist = np.sqrt(dx**2 + dy**2)
            bulge = 1.0 + 0.03 * (1 - dist**2)
            bulge = np.clip(bulge, 0.97, 1.03)

            cv2.ellipse(
                eye_region,
                (cx, cy),
                (int((roi_w // 2 - 5) * bulge), int((roi_h // 2 - 5) * bulge)),
                0,
                0,
                360,
                1.0,
                -1,
            )

            eye_region = cv2.GaussianBlur(eye_region, (5, 5), 0)

            openness = eye_data.get("openness", 1.0)
            lid_offset = int((1 - openness) * roi_h * 0.22)
            if lid_offset > 0:
                eye_region[:lid_offset, :] = 0
                eye_region = cv2.GaussianBlur(eye_region, (13, 13), 0)

            angle = eye_data.get("angle", 0)

            scale_x = 1.0 - abs(np.sin(np.radians(angle))) * 0.08
            new_w = int(roi_w * scale_x)
            offset = (roi_w - new_w) // 2

            color = color[:, offset : offset + new_w]
            alpha = alpha[:, offset : offset + new_w]
            eye_region = eye_region[:, offset : offset + new_w]

            if color.shape[1] != roi_w:
                color = cv2.resize(color, (roi_w, roi_h))
                alpha = cv2.resize(alpha, (roi_w, roi_h))
                eye_region = cv2.resize(eye_region, (roi_w, roi_h))

            roi = result[y1:y2, x1:x2].astype(np.float32)

            dark_base = roi * 0.5
            dark_base[:, :, 0] += 8 * intensity

            cy, cx = roi_h // 2, roi_w // 2
            y_grid, x_grid = np.ogrid[:roi_h, :roi_w]
            dx = (x_grid - cx) / (roi_w / 2)
            dy = (y_grid - cy) / (roi_h / 2)
            dist = np.sqrt(dx**2 + dy**2)
            sclera_shade = 1.0 - dist * 0.08
            sclera_shade = np.clip(sclera_shade, 0.92, 1.0)
            sclera_shade = sclera_shade[:, :, np.newaxis]

            blended_color = color.astype(np.float32) * sclera_shade

            combined = alpha * eye_region
            blended_color = blended_color * combined[:, :, np.newaxis]
            dark_base = dark_base * (1 - combined[:, :, np.newaxis])
            final = blended_color + dark_base

            final = np.clip(final, 0, 255).astype(np.uint8)

            final = self._cpu_blend(final, result[y1:y2, x1:x2], alpha * eye_region)

            result[y1:y2, x1:x2] = final

        if intensity > 0.15:
            result = self._apply_cinematic_grading(result, intensity)

        self._time += 0.016

        return result

    def _cpu_blend(
        self, fg: np.ndarray, bg: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """CPU fallback blending."""
        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        fg_float = fg.astype(np.float32)
        bg_float = bg.astype(np.float32)

        mask_3d = np.dstack([mask, mask, mask])

        result = (fg_float * mask_3d + bg_float * (1 - mask_3d)).astype(np.uint8)

        return result

    def _apply_cinematic_grading(
        self, frame: np.ndarray, intensity: float
    ) -> np.ndarray:
        """Apply subtle cinematic grading."""
        if intensity < 0.1:
            return frame

        result = frame.astype(np.float32)

        result = result * (0.96 + 0.04 * intensity)

        lift = 4 * intensity
        result = result + lift

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


def create_gpu_compositor(pack_path: Optional[str] = None) -> GPUEyeCompositor:
    """Factory function."""
    return GPUEyeCompositor(pack_path)
