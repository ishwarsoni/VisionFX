"""Reusable activation timer utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActivationTimer:
    """Simple accumulated timer with pause and reset support."""

    elapsed_ms: float = 0.0
    running: bool = False
    paused: bool = False

    def start(self, reset: bool = True) -> None:
        if reset:
            self.elapsed_ms = 0.0
        self.running = True
        self.paused = False

    def stop(self) -> None:
        self.running = False
        self.paused = False

    def pause(self) -> None:
        if self.running:
            self.paused = True

    def resume(self) -> None:
        if self.running:
            self.paused = False

    def reset(self) -> None:
        self.elapsed_ms = 0.0
        self.running = False
        self.paused = False

    def advance(self, delta_ms: float) -> None:
        if self.running and not self.paused and delta_ms > 0.0:
            self.elapsed_ms += delta_ms

    def set_elapsed(self, elapsed_ms: float) -> None:
        self.elapsed_ms = max(0.0, elapsed_ms)
