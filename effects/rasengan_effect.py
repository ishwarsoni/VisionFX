"""
Rasengan Effect - Clean implementation
"""

import math

import cv2
import numpy as np


class RasenganEffect:
    """Rasengan energy effect with clean hand-follow."""

    def __init__(self):
        self._position = (0, 0)
        self._active = False
        self._time = 0.0
        self._formation = 0.0

    def activate(self):
        """Activate Rasengan."""
        self._active = True
        self._formation = 0.0

    def deactivate(self):
        """Deactivate Rasengan."""
        self._active = False
        self._formation = 0.0
        self._position = (0, 0)

    def update(self, hand_center, dt, velocity=(0, 0)):
        """Update position - instant attachment to palm."""
        if not self._active:
            return

        # Instant position - no interpolation
        px, py = hand_center
        if px > 0 and py > 0:
            self._position = (px, py)

        # Formation buildup
        self._formation = min(1.0, self._formation + dt * 2.0)
        self._time += dt

    def render(self, frame, hand_center):
        """Render Rasengan effect."""
        if not self._active or self._formation < 0.05:
            return frame

        h, w = frame.shape[:2]
        cx, cy = int(self._position[0]), int(self._position[1])

        if cx <= 0 or cy <= 0:
            return frame

        radius = max(int(min(w, h) * 0.085 * self._formation), 26)

        overlay = np.zeros_like(frame)
        highlight = np.zeros_like(frame)
        spill = np.zeros_like(frame)

        # Base orb: layered blue-white body with a restrained core.
        for scale, color, thickness in (
            (1.50, (255, 185, 70), -1),
            (1.08, (255, 225, 120), -1),
            (0.70, (255, 245, 180), -1),
        ):
            cv2.circle(
                overlay,
                (cx, cy),
                max(1, int(radius * scale)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        cv2.circle(
            highlight,
            (cx, cy),
            max(1, int(radius * 0.34)),
            (255, 250, 245),
            -1,
            lineType=cv2.LINE_AA,
        )

        # Soft environmental blue spill around the hand.
        cv2.circle(
            spill,
            (cx, cy),
            max(1, int(radius * 1.75)),
            (255, 120, 20),
            -1,
            lineType=cv2.LINE_AA,
        )
        spill = cv2.GaussianBlur(spill, (0, 0), radius * 0.35)

        # Wispy outer shell: orbiting spiral strokes to mimic the reference cloud-like motion.
        num_wisps = 7
        for i in range(num_wisps):
            phase = self._time * 2.4 + i * (math.tau / num_wisps)
            points = []
            for step in range(14):
                t = step / 13.0
                wobble = math.sin(self._time * 4.0 + i * 1.7 + t * 8.0) * radius * 0.06
                orbit = radius * (0.62 + t * 0.82)
                angle = phase + t * math.tau * 1.35 + wobble * 0.01
                x = int(
                    cx
                    + math.cos(angle) * orbit
                    + math.cos(self._time * 1.8 + i + t * 6.0) * radius * 0.06
                )
                y = int(
                    cy
                    + math.sin(angle) * orbit
                    + math.sin(self._time * 1.6 + i * 0.9 + t * 5.0) * radius * 0.06
                )
                points.append((x, y))

            for start, end in zip(points[:-1], points[1:]):
                cv2.line(overlay, start, end, (255, 235, 150), 2, lineType=cv2.LINE_AA)

        # Inner spiral threads.
        for arm in range(4):
            angle_offset = self._time * 3.5 + arm * (math.tau / 4.0)
            for r in range(int(radius * 0.18), int(radius * 0.9), 4):
                angle = angle_offset + r * 0.045
                x = int(cx + math.cos(angle) * r)
                y = int(cy + math.sin(angle) * r)
                if 0 <= x < w and 0 <= y < h:
                    highlight[y, x] = (255, 255, 255)

        # Tight bright center.
        cv2.circle(
            highlight,
            (cx, cy),
            max(1, int(radius * 0.16)),
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        # Very subtle burst ring to echo the reference's soft perimeter.
        ring_radius = int(radius * (1.28 + 0.04 * math.sin(self._time * 5.0)))
        cv2.circle(
            overlay, (cx, cy), ring_radius, (255, 210, 110), 2, lineType=cv2.LINE_AA
        )

        frame = cv2.addWeighted(frame, 1.0, spill, 0.18, 0)
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.56, 0)
        frame = cv2.addWeighted(frame, 1.0, highlight, 0.42, 0)

        return frame


def create_rasengan():
    return RasenganEffect()
