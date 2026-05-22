"""Animated anime-style text cues for cinematic presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from presentation.activation_sequence import SequenceState
from presentation.presentation_data import PresentationContext


@dataclass
class TextCue:
    text: str
    kind: str
    duration_ms: float
    elapsed_ms: float = 0.0
    intensity: float = 1.0
    anchor: str = "center"
    glitch_seed: int = 0

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed_ms / max(1.0, self.duration_ms))

    @property
    def alive(self) -> bool:
        return self.elapsed_ms < self.duration_ms


class CinematicTextSystem:
    """Renders concise animated story cues."""

    def __init__(self, animation_ms: float):
        self.animation_ms = animation_ms
        self.cues: List[TextCue] = []

    def push(
        self,
        text: str,
        kind: str = "system",
        duration_ms: Optional[float] = None,
        intensity: float = 1.0,
        anchor: str = "center",
    ) -> None:
        self.cues.append(
            TextCue(
                text=text,
                kind=kind,
                duration_ms=duration_ms or self.animation_ms,
                intensity=intensity,
                anchor=anchor,
                glitch_seed=len(self.cues) * 31 + len(text),
            )
        )

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name in ("on_eye_lock", "on_stare_detected"):
            self.push("EYE CONTACT DETECTED", "lock", self.animation_ms, 1.0, "top")
        elif event_name == "on_activation_start":
            self.push(
                "POWER ACTIVATION",
                "activation",
                self.animation_ms * 1.15,
                1.0,
                "center",
            )
        elif event_name == "on_power_active":
            self.push(
                "SYSTEM TRANSFORMED",
                "transform",
                self.animation_ms * 1.25,
                1.0,
                "center",
            )
        elif event_name == "on_interrupt":
            self.push("SYSTEM UNSTABLE", "interrupt", self.animation_ms, 1.0, "center")
        elif event_name == "on_cooldown":
            self.push(
                "COOLDOWN ENGAGED", "cooldown", self.animation_ms * 0.8, 0.85, "bottom"
            )

    def update(self, delta_ms: float) -> None:
        for cue in self.cues:
            cue.elapsed_ms += max(0.0, delta_ms)
        self.cues = [cue for cue in self.cues if cue.alive]

    def render(
        self, frame: np.ndarray, context: PresentationContext, sequence: SequenceState
    ) -> np.ndarray:
        if not self.cues:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]
        for cue in self.cues:
            progress = cue.progress
            alpha = self._opacity(progress) * cue.intensity
            if alpha <= 0.01:
                continue

            base_y = (
                int(h * 0.18)
                if cue.anchor == "top"
                else int(h * 0.82) if cue.anchor == "bottom" else int(h * 0.48)
            )
            x = w // 2
            y = base_y + int(
                np.sin((context.timestamp_ms + cue.glitch_seed * 7) / 115.0)
                * (4 + sequence.corruption_intensity * 8)
            )
            jitter_x = int(
                np.sin((context.timestamp_ms + cue.glitch_seed * 13) / 40.0)
                * sequence.corruption_intensity
                * 7
            )

            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.82 + sequence.stage_progress * 0.08
            thickness = 2
            text_size, _ = cv2.getTextSize(cue.text, font, scale, thickness)
            tx = x - text_size[0] // 2 + jitter_x
            ty = y

            cv2.putText(
                overlay,
                cue.text,
                (tx + 2, ty + 2),
                font,
                scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            if cue.kind in ("activation", "transform", "interrupt") and progress > 0.2:
                color = (40, 255, 255) if cue.kind != "interrupt" else (0, 120, 255)
                cv2.putText(
                    overlay,
                    self._corrupt_text(cue.text, cue, context),
                    (tx, ty),
                    font,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    overlay,
                    cue.text,
                    (tx, ty),
                    font,
                    scale,
                    (0, 255, 210),
                    thickness,
                    cv2.LINE_AA,
                )

        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0.0)

    @staticmethod
    def _opacity(progress: float) -> float:
        if progress < 0.18:
            return progress / 0.18
        if progress > 0.82:
            return max(0.0, (1.0 - progress) / 0.18)
        return 1.0

    @staticmethod
    def _corrupt_text(text: str, cue: TextCue, context: PresentationContext) -> str:
        if cue.kind not in ("activation", "transform", "interrupt"):
            return text
        if int((context.timestamp_ms + cue.glitch_seed) / 70.0) % 3 == 0:
            return text.replace("A", "@", 1).replace("E", "3", 1)
        if int((context.timestamp_ms + cue.glitch_seed) / 90.0) % 2 == 0:
            return text.replace("O", "0")
        return text
