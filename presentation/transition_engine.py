"""Queued cinematic transition engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class TransitionRequest:
    """Single queued presentation transition."""

    name: str
    kind: str
    duration_ms: float
    start_value: float = 0.0
    end_value: float = 1.0
    interruptible: bool = True
    elapsed_ms: float = 0.0
    value: float = 0.0

    def advance(self, delta_ms: float) -> bool:
        self.elapsed_ms += max(0.0, delta_ms)
        duration = max(1.0, self.duration_ms)
        progress = min(1.0, self.elapsed_ms / duration)
        eased = progress * progress * (3.0 - 2.0 * progress)
        self.value = self.start_value + (self.end_value - self.start_value) * eased
        return progress >= 1.0


class TransitionEngine:
    """Queue-based transition handler for fades, pulses, zooms, and glitches."""

    def __init__(self):
        self.queue: Deque[TransitionRequest] = deque()
        self.current: Optional[TransitionRequest] = None
        self.current_value: float = 0.0
        self.active_kind: str = ""

    def queue_transition(
        self,
        name: str,
        kind: str,
        duration_ms: float,
        start_value: float = 0.0,
        end_value: float = 1.0,
        interruptible: bool = True,
    ) -> None:
        self.queue.append(
            TransitionRequest(
                name=name,
                kind=kind,
                duration_ms=max(1.0, duration_ms),
                start_value=start_value,
                end_value=end_value,
                interruptible=interruptible,
            )
        )

    def interrupt(
        self, name: str = "interrupt", kind: str = "fade", duration_ms: float = 120.0
    ) -> None:
        self.queue.clear()
        self.current = TransitionRequest(
            name=name,
            kind=kind,
            duration_ms=max(1.0, duration_ms),
            start_value=1.0,
            end_value=0.0,
        )
        self.current_value = 1.0
        self.active_kind = kind

    def update(self, delta_ms: float) -> float:
        if self.current is None and self.queue:
            self.current = self.queue.popleft()
            self.current_value = self.current.start_value
            self.active_kind = self.current.kind

        if self.current is None:
            self.current_value = 0.0
            self.active_kind = ""
            return self.current_value

        finished = self.current.advance(delta_ms)
        self.current_value = self.current.value
        if finished:
            self.current = None
        return self.current_value

    @property
    def queue_depth(self) -> int:
        return len(self.queue) + (1 if self.current is not None else 0)
