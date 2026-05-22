"""
GPU Chidori Renderer - ModernGL shader implementation (optional)
Dense violent lightning blade with procedural electric noise, bloom, and plasma rendering.
If ModernGL is not available, the module exposes a stub renderer.
"""

import cv2
import numpy as np

try:
    import moderngl

    MODERNGL_AVAILABLE = True
except Exception:
    MODERNGL_AVAILABLE = False


class GPUChidoriRenderer:
    def __init__(self):
        self.ctx = None
        self._available = False
        self._prog = None
        self._quad = None
        self._time = 0.0
        if MODERNGL_AVAILABLE:
            try:
                self.ctx = moderngl.create_standalone_context()
                self._init_shaders()
                self._available = True
            except Exception:
                self._available = False

    def _init_shaders(self):
        VERTEX = """#version 330
        in vec2 in_texcoord;
        in vec2 in_position;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_uv = in_texcoord;
        }
        """

        FRAGMENT = """#version 330
        in vec2 v_uv;
        out vec4 f_color;
        uniform sampler2D scene_tex;
        uniform vec2 hand_uv;
        uniform vec2 hand_dir;
        uniform float time;
        uniform float intensity;

        // Hash-based 2D noise
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        // Fractal Brownian Motion
        float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
            for (int i = 0; i < 5; i++) {
                v += a * noise(p);
                p = rot * p * 2.0;
                a *= 0.5;
            }
            return v;
        }

        // Lightning bolt function
        float lightning(vec2 uv, vec2 origin, vec2 dir, float t) {
            vec2 perp = vec2(-dir.y, dir.x);
            vec2 rel = uv - origin;
            float along = dot(rel, dir);
            float across = dot(rel, perp);

            // Only forward
            if (along < -0.02) return 0.0;

            // Jagged displacement
            float jag = fbm(vec2(along * 40.0 + t * 8.0, t * 3.0)) * 0.08;
            jag += fbm(vec2(along * 80.0 - t * 12.0, t * 5.0 + 1.0)) * 0.04;
            float displaced = across - jag;

            // Width narrows with distance
            float width = 0.025 * exp(-along * 4.0);
            float bolt = exp(-abs(displaced) / max(width, 0.001));
            bolt *= smoothstep(0.5, 0.0, along);

            return bolt * intensity;
        }

        void main() {
            vec2 uv = v_uv;
            vec4 scene = texture(scene_tex, uv);
            vec2 rel = uv - hand_uv;
            float dist = length(rel);

            float t = time;

            // === Dense electric core ===
            float core_pulse = 0.025 + 0.008 * sin(t * 25.0) + 0.005 * sin(t * 47.0);
            float core = exp(-dist / max(core_pulse, 0.001));
            float shell = exp(-dist / (core_pulse * 2.5)) * 0.7;

            // Turbulent noise in core
            float turb = fbm(uv * 120.0 + vec2(t * 6.0, t * 4.0));
            core *= (0.7 + 0.5 * turb);
            shell *= (0.6 + 0.6 * turb);

            // === Multiple lightning bolts ===
            float bolts = 0.0;
            for (int i = 0; i < 6; i++) {
                float angle_offset = float(i) * 1.047 + sin(t * 3.0 + float(i)) * 0.5;
                vec2 bolt_dir = vec2(
                    cos(atan(hand_dir.y, hand_dir.x) + angle_offset - 1.57),
                    sin(atan(hand_dir.y, hand_dir.x) + angle_offset - 1.57)
                );
                bolts += lightning(uv, hand_uv, bolt_dir, t + float(i) * 7.0) * 0.5;
            }

            // Forward blade bolts (denser)
            for (int i = 0; i < 4; i++) {
                float spread = (float(i) - 1.5) * 0.15;
                vec2 perp = vec2(-hand_dir.y, hand_dir.x);
                vec2 origin = hand_uv + perp * spread * 0.05;
                bolts += lightning(uv, origin, hand_dir, t + float(i) * 11.0) * 0.7;
            }

            // === Electric arcing noise ===
            float arc_noise = fbm(uv * 200.0 + vec2(t * 10.0, -t * 7.0));
            float arc = smoothstep(0.55, 0.9, arc_noise) * exp(-dist / 0.15) * intensity;

            // === Compose colors ===
            vec3 white_core = vec3(1.0, 1.0, 1.0) * core * intensity;
            vec3 blue_shell = vec3(0.0, 0.6, 1.0) * shell * intensity;
            vec3 blue_bolts = vec3(0.0, 0.8, 1.0) * bolts;
            vec3 arc_color = vec3(0.0, 0.9, 1.0) * arc;

            // Environment spill
            float spill = exp(-dist / 0.25) * intensity * 0.15;
            vec3 env_light = vec3(0.0, 0.5, 0.9) * spill;

            // Flickering
            float flicker = 0.7 + 0.3 * sin(t * 30.0 + hash(uv * 100.0) * 6.28);

            vec3 total = (white_core + blue_shell + blue_bolts + arc_color) * flicker + env_light;

            // Additive composite with bloom-like saturation
            vec3 outc = scene.rgb + total * 1.8;
            f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
        }
        """

        self._prog = self.ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self._vbo = self.ctx.buffer(
            np.array(
                [
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype="f4",
            ).tobytes()
        )
        self._vao = self.ctx.vertex_array(
            self._prog, [(self._vbo, "2f 2f", "in_texcoord", "in_position")]
        )

    def available(self):
        return self._available

    def render(
        self,
        frame: np.ndarray,
        hand_pos_px,
        hand_dir_vec=(0.0, -1.0),
        intensity=1.0,
        time=0.0,
    ) -> np.ndarray:
        """Render GPU overlay and composite onto frame."""
        if not self._available:
            return frame

        h, w = frame.shape[:2]
        try:
            tex = self.ctx.texture((w, h), 3, frame[:, :, ::-1].tobytes())
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.use()

            hand_uv = (
                float(hand_pos_px[0]) / float(w),
                float(hand_pos_px[1]) / float(h),
            )
            dir_n = np.array(hand_dir_vec, dtype=np.float32)
            dn = dir_n / max(1e-6, np.linalg.norm(dir_n))

            self._prog["scene_tex"].value = 0
            tex.use(0)
            self._prog["hand_uv"].value = tuple(hand_uv)
            self._prog["hand_dir"].value = (float(dn[0]), float(dn[1]))
            self._prog["time"].value = float(time)
            self._prog["intensity"].value = float(np.clip(intensity, 0.0, 1.0))

            self._vao.render(moderngl.TRIANGLE_STRIP)

            data = fbo.read(components=3)
            img = np.frombuffer(data, dtype="u1").reshape(h, w, 3)[:, :, ::-1]
            return img
        except Exception:
            return frame


class StubGPUChidoriRenderer:
    def __init__(self):
        self._available = False

    def available(self):
        return False

    def render(self, frame, *args, **kwargs):
        return frame


def create_gpu_chidori():
    if MODERNGL_AVAILABLE:
        return GPUChidoriRenderer()
    return StubGPUChidoriRenderer()
