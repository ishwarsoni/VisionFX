"""Validation windows and hysteresis for activation transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationWindow:
    """Temporal validation window with hysteresis."""

    threshold_ms: float
    decay_factor: float = 0.5
    accumulated_ms: float = 0.0

    def update(self, condition: bool, delta_ms: float) -> bool:
        if condition:
            self.accumulated_ms = min(
                self.threshold_ms, self.accumulated_ms + max(0.0, delta_ms)
            )
        else:
            self.accumulated_ms = max(
                0.0, self.accumulated_ms - max(0.0, delta_ms) * self.decay_factor
            )
        return self.is_stable

    def reset(self) -> None:
        self.accumulated_ms = 0.0

    @property
    def is_stable(self) -> bool:
        return self.accumulated_ms >= self.threshold_ms


class ActivationTransitionManager:
    """Manages validation windows for face, stare, and interruption states."""

    def __init__(self, stability_window_ms: float, interruption_threshold_ms: float):
        self.face_window = ValidationWindow(stability_window_ms)
        self.stare_window = ValidationWindow(stability_window_ms)
        self.interrupt_window = ValidationWindow(interruption_threshold_ms)
        self.last_interrupt_reason = ""

    def update_face(self, is_valid: bool, delta_ms: float) -> bool:
        return self.face_window.update(is_valid, delta_ms)

    def update_stare(self, is_valid: bool, delta_ms: float) -> bool:
        return self.stare_window.update(is_valid, delta_ms)

    def update_interrupt(
        self, is_valid: bool, delta_ms: float, reason: str = ""
    ) -> bool:
        stable = self.interrupt_window.update(is_valid, delta_ms)
        if stable:
            self.last_interrupt_reason = reason
        elif not is_valid:
            self.last_interrupt_reason = ""
        return stable

    def reset(self) -> None:
        self.face_window.reset()
        self.stare_window.reset()
        self.interrupt_window.reset()
        self.last_interrupt_reason = ""
