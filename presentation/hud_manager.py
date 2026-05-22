"""Anime-inspired cinematic HUD overlays."""

from __future__ import annotations

import math

import cv2
import numpy as np

from presentation.activation_sequence import SequenceState
from presentation.presentation_data import PresentationContext


class HUDManager:
    """Renders minimal but dramatic HUD and reticle elements."""

    def __init__(self, hud_intensity: float):
        self.hud_intensity = hud_intensity
        self.current_alpha = 0.0

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name in ("on_eye_lock", "on_stare_detected"):
            self.current_alpha = max(self.current_alpha, 0.75)
        elif event_name == "on_activation_start":
            self.current_alpha = max(self.current_alpha, 0.9)
        elif event_name == "on_interrupt":
            self.current_alpha = max(self.current_alpha, 0.8)
        elif event_name == "on_cooldown":
            self.current_alpha = max(self.current_alpha, 0.55)

    def update(self, context: PresentationContext, sequence: SequenceState) -> None:
        target = max(
            sequence.eye_lock_intensity,
            sequence.buildup_intensity,
            sequence.transform_intensity,
            sequence.interruption_intensity,
        )
        target *= self.hud_intensity
        self.current_alpha += (target - self.current_alpha) * min(
            1.0, context.delta_ms / 160.0
        )

    def render(
        self, frame: np.ndarray, context: PresentationContext, sequence: SequenceState
    ) -> np.ndarray:
        if self.current_alpha <= 0.01:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]
        overlay = result.copy()

        self._draw_cinematic_bars(overlay, w, h, sequence)
        self._draw_activation_meter(overlay, w, h, context, sequence)
        self._draw_reticle(overlay, w, h, context, sequence)
        self._draw_status_text(overlay, w, h, context, sequence)

        return cv2.addWeighted(
            result,
            1.0 - self.current_alpha * 0.7,
            overlay,
            self.current_alpha * 0.7,
            0.0,
        )

    def _draw_cinematic_bars(
        self, frame: np.ndarray, width: int, height: int, sequence: SequenceState
    ) -> None:
        bar_height = int(height * (0.06 + sequence.transform_intensity * 0.05))
        if bar_height <= 0:
            return
        cv2.rectangle(frame, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, height - bar_height), (width, height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, bar_height), (0, 255, 210), 1)
        cv2.rectangle(
            frame, (0, height - bar_height), (width, height), (0, 255, 210), 1
        )

    def _draw_activation_meter(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        context: PresentationContext,
        sequence: SequenceState,
    ) -> None:
        meter_w = int(width * 0.26)
        meter_h = 12
        x = 26
        y = height - 42
        progress = max(0.0, min(1.0, context.activation_progress / 100.0))
        cv2.putText(
            frame,
            "POWER LEVEL",
            (x, y - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            (0, 255, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x, y), (x + meter_w, y + meter_h), (20, 20, 20), -1)
        fill = int(meter_w * progress)
        fill_color = (
            (0, 255, 210) if sequence.stage.value != "INTERRUPTED" else (0, 120, 255)
        )
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + meter_h), fill_color, -1)
        cv2.rectangle(frame, (x, y), (x + meter_w, y + meter_h), (0, 255, 210), 1)
        for tick in range(1, 5):
            tx = x + int(meter_w * tick / 5)
            cv2.line(frame, (tx, y - 2), (tx, y + meter_h + 2), (0, 255, 210), 1)

    def _draw_reticle(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        context: PresentationContext,
        sequence: SequenceState,
    ) -> None:
        if context.tracking_data is None or not context.tracking_data.face.detected:
            return
        cx = int(context.tracking_data.face.center[0])
        cy = int(context.tracking_data.face.center[1])
        if cx <= 0 or cy <= 0:
            return
        size = int(42 + sequence.eye_lock_intensity * 18)
        pulse = 1.0 + math.sin(context.timestamp_ms / 180.0) * 0.06
        color = (
            (0, 255, 210) if sequence.stage.value != "INTERRUPTED" else (0, 120, 255)
        )
        cv2.circle(frame, (cx, cy), int(size * pulse), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - size, cy), (cx - size + 12, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + size - 12, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - size), (cx, cy - size + 12), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + size - 12), (cx, cy + size), color, 1, cv2.LINE_AA)

    def _draw_status_text(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        context: PresentationContext,
        sequence: SequenceState,
    ) -> None:
        label = sequence.stage.value.replace("_", " ")
        text = f"{label} | {context.activation_progress:.0f}%"
        color = (
            (0, 255, 210) if sequence.stage.value != "INTERRUPTED" else (0, 120, 255)
        )
        cv2.putText(
            frame, text, (26, 42), cv2.FONT_HERSHEY_DUPLEX, 0.56, color, 1, cv2.LINE_AA
        )
