"""Volume transition helpers for cinematic crossfades."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    scaled = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


@dataclass
class VolumeEnvelope:
    current: float = 0.0
    target: float = 0.0
    start_value: float = 0.0
    duration_ms: float = 0.0
    elapsed_ms: float = 0.0
    active: bool = False

    def set_target(self, target: float, duration_ms: float) -> None:
        self.start_value = self.current
        self.target = clamp(target)
        self.duration_ms = max(0.0, duration_ms)
        self.elapsed_ms = 0.0
        self.active = (
            self.duration_ms > 0.0 and abs(self.target - self.current) > 0.0001
        )
        if self.duration_ms == 0.0:
            self.current = self.target
            self.active = False

    def update(self, delta_ms: float) -> float:
        if not self.active:
            self.current = self.target
            return self.current

        self.elapsed_ms = min(self.duration_ms, self.elapsed_ms + max(0.0, delta_ms))
        progress = smoothstep(0.0, 1.0, self.elapsed_ms / max(1.0, self.duration_ms))
        self.current = self.start_value + (self.target - self.start_value) * progress

        if self.elapsed_ms >= self.duration_ms:
            self.current = self.target
            self.active = False

        return self.current


def crossfade(
    out_volume: float, in_volume: float, progress: float
) -> tuple[float, float]:
    mix = smoothstep(0.0, 1.0, progress)
    return out_volume * (1.0 - mix), in_volume * mix
