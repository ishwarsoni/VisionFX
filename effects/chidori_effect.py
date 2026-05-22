"""
Naruto-style Chidori Lightning Blade - Dense violent electricity.
Covers entire hand with crawling lightning, creates a dense electric core,
emits violent branching arcs, and illuminates the environment.
"""

import math
import random

import cv2
import numpy as np

try:
    from effects.gpu_chidori import create_gpu_chidori
except Exception:
    create_gpu_chidori = None


class ChidoriEffect:
    """TRUE Naruto-style Chidori with dense violent lightning."""

    def __init__(self):
        self._position = (0, 0)
        self._active = False
        self._time = 0.0
        self._formation = 0.0
        self._velocity = (0, 0)
        self._last_velocity = (0, 0)
        self._flicker_state = 0.0
        self._arc_seed = 0
        self._particle_pool = []
        self._trail = []
        self._trail_max = 12
        self._gpu = create_gpu_chidori() if create_gpu_chidori else None
        self.use_rasengan_scale = False
        self._last_points = None

        # Full-hand tracking
        self._hand_landmarks_px = None
        self._blade_direction = np.array([0.0, -1.0], dtype=np.float32)
        self._hand_rotation = 0.0
        self._hand_scale_px = 32.0
        self._anchor_point = (0.0, 0.0)
        self._crawling_arcs = []
        self._blade_noise_seed = random.randint(0, 10000)

        # Precompute blue palette for speed
        self._core_white = (255, 255, 255)
        self._blue_bright = (255, 230, 0)  # BGR bright cyan-blue
        self._blue_mid = (255, 180, 0)  # BGR medium blue
        self._blue_deep = (240, 140, 0)  # BGR deep blue
        self._blue_glow = (220, 120, 0)  # BGR subtle glow

    # ── activation / deactivation ──────────────────────────────

    def activate(self):
        self._active = True
        self._formation = 0.0
        self._flicker_state = 1.0
        self._arc_seed = random.randint(0, 10000)
        self._blade_noise_seed = random.randint(0, 10000)
        self._crawling_arcs = []

    def deactivate(self):
        self._active = False
        self._formation = 0.0
        self._position = (0, 0)
        self._particle_pool = []
        self._trail = []
        self._crawling_arcs = []
        self._hand_landmarks_px = None
        self._anchor_point = (0.0, 0.0)

    # ── hand geometry ──────────────────────────────────────────

    def _compute_hand_geometry(self, points, frame_shape=None):
        if not points or not hasattr(points, "landmark"):
            return
        if frame_shape is None:
            frame_shape = (480, 640)
        h, w = frame_shape
        lms = points.landmark

        self._hand_landmarks_px = []
        for lm in lms:
            self._hand_landmarks_px.append((lm.x * w, lm.y * h))

        if len(lms) >= 10:
            wrist = np.array(self._hand_landmarks_px[0], dtype=np.float32)
            middle_tip = np.array(self._hand_landmarks_px[12], dtype=np.float32)
            palm_mcp = np.array(
                [
                    self._hand_landmarks_px[5],
                    self._hand_landmarks_px[9],
                    self._hand_landmarks_px[13],
                    self._hand_landmarks_px[17],
                ],
                dtype=np.float32,
            )

            forward = middle_tip - wrist
            norm = np.linalg.norm(forward)
            if norm > 1e-6:
                self._blade_direction = forward / norm
            else:
                self._blade_direction = np.array([0.0, -1.0], dtype=np.float32)

            palm_center = np.mean(palm_mcp, axis=0)
            anchor = palm_center + self._blade_direction * 0.12 * max(1.0, norm)
            self._anchor_point = (float(anchor[0]), float(anchor[1]))
            self._position = self._anchor_point

            palm_width = float(np.linalg.norm(palm_mcp[0] - palm_mcp[3]))
            wrist_middle = float(np.linalg.norm(wrist - middle_tip))
            bbox_min = np.min(
                np.array(self._hand_landmarks_px, dtype=np.float32), axis=0
            )
            bbox_max = np.max(
                np.array(self._hand_landmarks_px, dtype=np.float32), axis=0
            )
            bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
            self._hand_scale_px = max(
                10.0, float(np.mean([palm_width, wrist_middle * 0.8, bbox_diag * 0.6]))
            )
            self._hand_rotation = float(
                math.atan2(self._blade_direction[1], self._blade_direction[0])
            )

    # ── update ─────────────────────────────────────────────────

    def update(self, hand_center, dt, velocity=(0, 0), points=None, frame_shape=None):
        if not self._active:
            return
        px, py = hand_center
        if px > 0 and py > 0:
            self._last_velocity = self._velocity
            self._velocity = velocity
            self._position = (px, py)

        self._formation = min(1.0, self._formation + dt * 5.0)
        self._time += dt

        # Violent rapid flicker
        self._flicker_state = 0.4 + 0.6 * (
            0.5 + 0.5 * math.sin(self._time * 30.0 + random.random() * 4.0)
        )

        if points is not None and frame_shape is not None:
            self._compute_hand_geometry(points, frame_shape)

        self._update_particles(dt)

        if hand_center and hand_center[0] > 0 and hand_center[1] > 0:
            self._trail.insert(0, (hand_center[0], hand_center[1], self._time))
            if len(self._trail) > self._trail_max:
                self._trail.pop()

    # ── lightning arc generation ───────────────────────────────

    def _generate_arc(self, sx, sy, ex, ey, depth=0, max_depth=3):
        """Recursive branching lightning bolt."""
        if depth > max_depth:
            return [(sx, sy)]

        mx = (sx + ex) * 0.5
        my = (sy + ey) * 0.5
        dx = ex - sx
        dy = ey - sy
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        nx = -dy / dist
        ny = dx / dist

        jitter = dist * (0.35 - 0.08 * depth)
        mx += nx * random.uniform(-jitter, jitter)
        my += ny * random.uniform(-jitter, jitter)

        pts = self._generate_arc(sx, sy, mx, my, depth + 1, max_depth)
        pts.extend(self._generate_arc(mx, my, ex, ey, depth + 1, max_depth))

        # Branching
        if depth < max_depth - 1 and random.random() < 0.45:
            ba = random.uniform(-math.pi, math.pi)
            bl = dist * random.uniform(0.3, 0.6)
            bx = mx + math.cos(ba) * bl
            by = my + math.sin(ba) * bl
            pts.extend(self._generate_arc(mx, my, bx, by, depth + 1, max_depth))

        return pts

    def _draw_arc_line(
        self, target, pts, core_thick, glow_thick, core_color, glow_color
    ):
        """Draw a polyline arc with core + glow layers."""
        for i in range(len(pts) - 1):
            p1 = (int(pts[i][0]), int(pts[i][1]))
            p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
            # Outer glow first
            cv2.line(target, p1, p2, glow_color, glow_thick, lineType=cv2.LINE_AA)
            # Inner core
            cv2.line(target, p1, p2, core_color, core_thick, lineType=cv2.LINE_AA)

    # ── particles ──────────────────────────────────────────────

    def _update_particles(self, dt):
        if not self._active:
            return

        motion_mag = math.sqrt(self._velocity[0] ** 2 + self._velocity[1] ** 2)
        spawn_rate = int(30 + 30 * self._formation + 20 * min(1.0, motion_mag / 50.0))

        for _ in range(spawn_rate):
            if self._hand_landmarks_px and len(self._hand_landmarks_px) > 0:
                ref = random.choice(
                    self._hand_landmarks_px[: min(9, len(self._hand_landmarks_px))]
                )
                px = ref[0] + random.uniform(-18, 18)
                py = ref[1] + random.uniform(-18, 18)
            else:
                px = self._position[0] + random.uniform(-15, 15)
                py = self._position[1] + random.uniform(-15, 15)

            if random.random() < 0.35:
                angle = math.atan2(self._blade_direction[1], self._blade_direction[0])
                speed = random.uniform(250.0, 400.0)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
            else:
                angle = random.random() * math.tau
                speed = random.uniform(150.0, 320.0)
                vx = math.cos(angle) * speed + self._velocity[0] * 0.5
                vy = math.sin(angle) * speed + self._velocity[1] * 0.5

            lifetime = random.uniform(0.05, 0.18)
            self._particle_pool.append(
                {
                    "x": px,
                    "y": py,
                    "vx": vx,
                    "vy": vy,
                    "life": lifetime,
                    "max_life": lifetime,
                    "intensity": random.uniform(0.7, 1.0),
                    "size": random.uniform(1.5, 4.0),
                }
            )

        dead = []
        for i, p in enumerate(self._particle_pool):
            p["life"] -= dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            dx = self._position[0] - p["x"]
            dy = self._position[1] - p["y"]
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            p["vx"] += (dx / dist) * 80.0 * dt
            p["vy"] += (dy / dist) * 80.0 * dt
            if p["life"] <= 0:
                dead.append(i)

        for i in reversed(dead):
            self._particle_pool.pop(i)

        # Cap pool
        if len(self._particle_pool) > 600:
            self._particle_pool = self._particle_pool[-600:]

    # ── render entry ───────────────────────────────────────────

    def render(self, frame, hand_center, points=None):
        """Render dense violent Chidori lightning."""
        if not self._active or self._formation < 0.05:
            return frame

        h, w = frame.shape[:2]
        cx, cy = int(self._position[0]), int(self._position[1])
        if cx <= 0 or cy <= 0:
            return frame

        if points is not None:
            self._compute_hand_geometry(points, (h, w))
            cx, cy = int(self._position[0]), int(self._position[1])

        # Working layers
        electric = np.zeros((h, w, 3), dtype=np.uint8)
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        radius = max(int(self._hand_scale_px * (0.32 + 0.36 * self._formation)), 14)
        flk = self._flicker_state
        form = self._formation

        # ─── 1. DENSE ELECTRIC CORE ───────────────────────────
        self._render_electric_core(electric, glow_layer, cx, cy, radius, flk, form)

        # ─── 2. FULL-HAND ELECTRICITY ─────────────────────────
        has_hand = self._hand_landmarks_px and len(self._hand_landmarks_px) >= 21
        if has_hand:
            self._render_hand_electricity(electric, radius, flk, form)
        else:
            self._render_radial_electricity(electric, cx, cy, radius, flk, form, w, h)

        # ─── 3. VIOLENT BRANCHING ARCS ────────────────────────
        self._render_violent_arcs(electric, cx, cy, radius, flk, form, w, h)

        # ─── 4. PARTICLES ─────────────────────────────────────
        self._render_particles(electric)

        # ─── 5. HAND ILLUMINATION ─────────────────────────────
        if has_hand:
            self._render_hand_glow(glow_layer, radius, flk)

        # ─── 6. ENVIRONMENT LIGHT SPILL ───────────────────────
        spill_r = int(radius * 3.5 * form)
        cv2.circle(
            glow_layer, (cx, cy), spill_r, self._blue_mid, -1, lineType=cv2.LINE_AA
        )

        # ─── COMPOSITE ────────────────────────────────────────
        # Blur glow layer for bloom
        ks = max(3, (int(radius * 0.8) | 1))
        glow_blurred = cv2.GaussianBlur(glow_layer, (ks, ks), 0)

        # Strong additive compositing
        frame = cv2.add(
            frame,
            cv2.multiply(
                glow_blurred,
                (0.25 * form, 0.25 * form, 0.25 * form, 0),
                dtype=cv2.CV_8U,
            ),
        )
        frame = cv2.add(
            frame,
            cv2.multiply(
                electric, (0.85 * flk, 0.85 * flk, 0.85 * flk, 0), dtype=cv2.CV_8U
            ),
        )

        # Electric bloom pass on the electric layer itself
        bloom_ks = max(3, (int(radius * 0.5) | 1))
        bloom = cv2.GaussianBlur(electric, (bloom_ks, bloom_ks), 0)
        frame = cv2.add(
            frame,
            cv2.multiply(bloom, (0.4 * flk, 0.4 * flk, 0.4 * flk, 0), dtype=cv2.CV_8U),
        )

        # ─── SCREEN EFFECTS ───────────────────────────────────
        if flk > 0.7 and random.random() < 0.25:
            shake = random.randint(-3, 3)
            frame = np.roll(frame, shake, axis=1)

        if flk > 0.6:
            exposure = 1.0 + (flk - 0.6) * 0.5
            frame = cv2.convertScaleAbs(frame, alpha=exposure, beta=10)

        return frame

    # ── dense electric core ────────────────────────────────────

    def _render_electric_core(self, target, glow, cx, cy, radius, flk, form):
        """Render the white-hot center with blue plasma shell."""
        # Unstable pulsing core size
        pulse = (
            0.8
            + 0.3 * math.sin(self._time * 25.0)
            + 0.15 * math.sin(self._time * 47.0 + 1.3)
        )
        core_r = max(4, int(radius * 0.45 * pulse * form))
        shell_r = max(6, int(radius * 0.7 * pulse * form))
        outer_r = max(8, int(radius * 1.0 * form))

        # Deep blue outer plasma shell on glow
        cv2.circle(glow, (cx, cy), outer_r, self._blue_deep, -1, lineType=cv2.LINE_AA)
        # Blue plasma shell
        cv2.circle(target, (cx, cy), shell_r, self._blue_mid, -1, lineType=cv2.LINE_AA)
        # Bright blue inner
        cv2.circle(
            target,
            (cx, cy),
            int(core_r * 1.2),
            self._blue_bright,
            -1,
            lineType=cv2.LINE_AA,
        )
        # White-hot center
        cv2.circle(target, (cx, cy), core_r, self._core_white, -1, lineType=cv2.LINE_AA)

        # Turbulence sparks inside core
        for _ in range(int(8 * form)):
            angle = random.random() * math.tau
            r = random.uniform(0, core_r * 1.3)
            sx = int(cx + math.cos(angle) * r)
            sy = int(cy + math.sin(angle) * r)
            cv2.circle(target, (sx, sy), random.randint(1, 3), self._core_white, -1)

    # ── full-hand electricity ──────────────────────────────────

    def _render_hand_electricity(self, target, radius, flk, form):
        """Dense electricity covering every finger and palm."""
        pts = self._hand_landmarks_px
        h, w = target.shape[:2]

        # Finger paths: MCP -> PIP -> DIP -> TIP
        finger_chains = [
            [1, 2, 3, 4],  # Thumb
            [5, 6, 7, 8],  # Index
            [9, 10, 11, 12],  # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20],  # Pinky
        ]

        thick_core = max(2, int(3.5 * flk * form))
        thick_glow = max(4, int(7.0 * flk * form))

        # Multiple arc passes for density
        for _pass in range(3):
            for chain in finger_chains:
                for j in range(len(chain) - 1):
                    i1, i2 = chain[j], chain[j + 1]
                    if i1 < len(pts) and i2 < len(pts):
                        arc = self._generate_arc(
                            pts[i1][0], pts[i1][1], pts[i2][0], pts[i2][1], max_depth=2
                        )
                        self._draw_arc_line(
                            target,
                            arc,
                            thick_core,
                            thick_glow,
                            self._core_white,
                            self._blue_bright,
                        )

        # Cross-finger arcs (chaotic connections)
        cross_pairs = [
            (4, 8),
            (8, 12),
            (12, 16),
            (16, 20),
            (4, 12),
            (8, 16),
            (5, 17),
            (0, 9),
            (0, 5),
            (0, 17),
        ]
        for i1, i2 in cross_pairs:
            if random.random() < 0.6 and i1 < len(pts) and i2 < len(pts):
                arc = self._generate_arc(
                    pts[i1][0], pts[i1][1], pts[i2][0], pts[i2][1], max_depth=2
                )
                self._draw_arc_line(
                    target,
                    arc,
                    max(1, thick_core - 1),
                    max(3, thick_glow - 2),
                    self._core_white,
                    self._blue_mid,
                )

        # Palm fill: dense arcs across palm surface
        palm_indices = [0, 5, 9, 13, 17]
        for _pass in range(2):
            for i in range(len(palm_indices)):
                for j in range(i + 1, len(palm_indices)):
                    if random.random() < 0.5:
                        i1, i2 = palm_indices[i], palm_indices[j]
                        arc = self._generate_arc(
                            pts[i1][0], pts[i1][1], pts[i2][0], pts[i2][1], max_depth=2
                        )
                        self._draw_arc_line(
                            target,
                            arc,
                            max(1, thick_core - 1),
                            max(2, thick_glow - 2),
                            self._blue_bright,
                            self._blue_mid,
                        )

        # Fingertip spark bursts
        tip_indices = [4, 8, 12, 16, 20]
        for ti in tip_indices:
            if ti < len(pts):
                tx, ty = pts[ti]
                for _ in range(int(4 * form)):
                    angle = random.random() * math.tau
                    length = random.uniform(8, 25) * form
                    ex = tx + math.cos(angle) * length
                    ey = ty + math.sin(angle) * length
                    arc = self._generate_arc(tx, ty, ex, ey, max_depth=1)
                    self._draw_arc_line(
                        target, arc, 1, 2, self._core_white, self._blue_bright
                    )

    # ── fallback radial electricity ────────────────────────────

    def _render_radial_electricity(self, target, cx, cy, radius, flk, form, w, h):
        """Dense radial arcs when no hand landmarks available."""
        num_arcs = 10 + int(6 * form)
        thick_core = max(2, int(3.0 * flk * form))
        thick_glow = max(4, int(6.0 * flk * form))

        for i in range(num_arcs):
            angle = (i / num_arcs) * math.tau + random.uniform(-0.3, 0.3)
            dist = radius * (1.8 + 2.0 * form)
            ex = cx + math.cos(angle) * dist
            ey = cy + math.sin(angle) * dist
            ex = max(0, min(w - 1, ex))
            ey = max(0, min(h - 1, ey))
            arc = self._generate_arc(float(cx), float(cy), ex, ey, max_depth=3)
            self._draw_arc_line(
                target, arc, thick_core, thick_glow, self._core_white, self._blue_bright
            )

    # ── violent branching arcs ─────────────────────────────────

    def _render_violent_arcs(self, target, cx, cy, radius, flk, form, w, h):
        """Long violent arcs shooting outward from the core."""
        num_long = 5 + int(5 * form)
        thick_core = max(2, int(3.5 * flk))
        thick_glow = max(5, int(8.0 * flk))

        for i in range(num_long):
            # Prefer blade direction but with spread
            base_angle = math.atan2(self._blade_direction[1], self._blade_direction[0])
            angle = base_angle + random.uniform(-1.5, 1.5)
            length = radius * random.uniform(1.5, 4.0) * form
            ex = cx + math.cos(angle) * length
            ey = cy + math.sin(angle) * length
            ex = max(0, min(w - 1, ex))
            ey = max(0, min(h - 1, ey))

            arc = self._generate_arc(float(cx), float(cy), ex, ey, max_depth=3)
            self._draw_arc_line(
                target, arc, thick_core, thick_glow, self._core_white, self._blue_bright
            )

        # High-voltage burst arcs (occasional very long ones)
        if random.random() < 0.4 * form:
            angle = random.random() * math.tau
            length = radius * random.uniform(4.0, 7.0) * form
            ex = cx + math.cos(angle) * length
            ey = cy + math.sin(angle) * length
            ex = max(0, min(w - 1, ex))
            ey = max(0, min(h - 1, ey))
            arc = self._generate_arc(float(cx), float(cy), ex, ey, max_depth=4)
            self._draw_arc_line(
                target,
                arc,
                max(3, int(4 * flk)),
                max(6, int(10 * flk)),
                self._core_white,
                self._blue_mid,
            )

    # ── hand glow illumination ─────────────────────────────────

    def _render_hand_glow(self, target, radius, flk):
        """Blue-white illumination across entire hand."""
        pts = self._hand_landmarks_px
        hand_pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(hand_pts)
        cv2.fillConvexPoly(target, hull, self._blue_bright)

        # Bright tips
        for ti in [4, 8, 12, 16, 20]:
            if ti < len(pts):
                tx, ty = int(pts[ti][0]), int(pts[ti][1])
                cv2.circle(
                    target, (tx, ty), max(4, int(radius * 0.3)), self._core_white, -1
                )

        # Wrist glow
        wx, wy = int(pts[0][0]), int(pts[0][1])
        cv2.circle(target, (wx, wy), max(4, int(radius * 0.25)), self._blue_bright, -1)

    # ── particles ──────────────────────────────────────────────

    def _render_particles(self, target):
        h, w = target.shape[:2]
        for p in self._particle_pool:
            px, py = int(p["x"]), int(p["y"])
            if 0 <= px < w and 0 <= py < h:
                fade = p["life"] / max(p["max_life"], 0.01)
                size = max(1, int(p["size"] * fade))
                bright = fade * p["intensity"] * self._flicker_state
                if bright > 0.3:
                    # Outer blue glow
                    cv2.circle(target, (px, py), size + 2, self._blue_mid, -1)
                    # Inner white spark
                    cv2.circle(target, (px, py), size, self._core_white, -1)


def create_chidori():
    return ChidoriEffect()
