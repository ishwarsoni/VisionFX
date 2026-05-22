"""Cooldown lockout management."""

from __future__ import annotations

from dataclasses import dataclass, field

from activation.activation_timer import ActivationTimer


@dataclass
class CooldownManager:
    """Manages activation lockout periods."""

    timer: ActivationTimer = field(default_factory=ActivationTimer)
    duration_ms: float = 0.0

    def start(self, duration_ms: float) -> None:
        self.duration_ms = max(0.0, duration_ms)
        self.timer.start(reset=True)

    def stop(self) -> None:
        self.timer.stop()

    def reset(self) -> None:
        self.duration_ms = 0.0
        self.timer.reset()

    def advance(self, delta_ms: float) -> None:
        self.timer.advance(delta_ms)

    @property
    def is_active(self) -> bool:
        return self.timer.running and self.remaining_ms > 0.0

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.duration_ms - self.timer.elapsed_ms)

    @property
    def is_complete(self) -> bool:
        return self.timer.running and self.timer.elapsed_ms >= self.duration_ms
