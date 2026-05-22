"""Shared audio data structures for Phase 6."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AudioLayer(str, Enum):
    AMBIENCE = "ambience"
    ACTIVATION = "activation"
    GLITCH = "glitch"
    TRANSITION = "transition"
    UI = "ui"


@dataclass
class AudioCue:
    cue_id: str
    layer: AudioLayer
    event_name: str
    duration_ms: float
    volume: float = 1.0
    pan: float = 0.0
    loop: bool = False
    priority: int = 0
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    sample_rate: int = 44100
    samples: Any = None
    sound_path: str = ""
    description: str = ""
    timestamp_ms: float = 0.0
    scheduled_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return self.description or self.cue_id


@dataclass
class LayerTarget:
    layer: AudioLayer
    cue: Optional[AudioCue] = None
    volume: float = 0.0
    pan: float = 0.0
    fade_ms: float = 0.0
    active: bool = True


@dataclass
class AudioPlan:
    cues: List[AudioCue] = field(default_factory=list)
    layer_targets: Dict[AudioLayer, LayerTarget] = field(default_factory=dict)
    mood: str = "idle"
    notes: List[str] = field(default_factory=list)

    def merge(self, other: "AudioPlan") -> None:
        self.cues.extend(other.cues)
        self.notes.extend(other.notes)
        if other.mood != "idle":
            self.mood = other.mood
        self.layer_targets.update(other.layer_targets)


@dataclass
class AudioEventContext:
    event_name: str = ""
    timestamp_ms: float = 0.0
    state: str = "IDLE"
    progress: float = 0.0
    stare_duration_ms: float = 0.0
    interruption_duration_ms: float = 0.0
    cooldown_remaining_ms: float = 0.0
    activation_count: int = 0
    reason: str = ""
    intensity: float = 0.0
    eye_lock_strength: float = 0.0
    glitch_intensity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls, event_name: str, payload: Dict[str, Any]
    ) -> "AudioEventContext":
        snapshot = payload.get("snapshot") or {}
        progress = float(
            snapshot.get("activation_progress", payload.get("progress", 0.0))
        )
        stare_duration_ms = float(snapshot.get("stare_duration_ms", 0.0))
        interruption_duration_ms = float(snapshot.get("interruption_duration_ms", 0.0))
        cooldown_remaining_ms = float(snapshot.get("cooldown_remaining_ms", 0.0))
        intensity = max(0.0, min(1.0, progress / 100.0))
        eye_lock_strength = max(0.0, min(1.0, stare_duration_ms / 1500.0))

        return cls(
            event_name=event_name,
            timestamp_ms=_current_time_ms(),
            state=str(snapshot.get("state", payload.get("state", "IDLE"))),
            progress=progress,
            stare_duration_ms=stare_duration_ms,
            interruption_duration_ms=interruption_duration_ms,
            cooldown_remaining_ms=cooldown_remaining_ms,
            activation_count=int(
                snapshot.get("activation_count", payload.get("activation_count", 0))
            ),
            reason=str(payload.get("reason", "")),
            intensity=intensity,
            eye_lock_strength=eye_lock_strength,
            glitch_intensity=max(intensity, min(1.0, eye_lock_strength + 0.2)),
            details=dict(snapshot.get("details") or {}),
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: Any, event_name: str = "on_activation_progress"
    ) -> "AudioEventContext":
        state_value = getattr(snapshot.state, "value", snapshot.state)
        progress = float(getattr(snapshot, "activation_progress", 0.0))
        stare_duration_ms = float(getattr(snapshot, "stare_duration_ms", 0.0))
        interruption_duration_ms = float(
            getattr(snapshot, "interruption_duration_ms", 0.0)
        )
        cooldown_remaining_ms = float(getattr(snapshot, "cooldown_remaining_ms", 0.0))
        intensity = max(0.0, min(1.0, progress / 100.0))

        return cls(
            event_name=event_name,
            timestamp_ms=_current_time_ms(),
            state=str(state_value),
            progress=progress,
            stare_duration_ms=stare_duration_ms,
            interruption_duration_ms=interruption_duration_ms,
            cooldown_remaining_ms=cooldown_remaining_ms,
            activation_count=int(getattr(snapshot, "activation_count", 0)),
            reason=str(getattr(snapshot, "transition_reason", "")),
            intensity=intensity,
            eye_lock_strength=max(0.0, min(1.0, stare_duration_ms / 1500.0)),
            glitch_intensity=max(intensity, 0.15),
            details=dict(getattr(snapshot, "details", {}) or {}),
        )


@dataclass
class AudioSnapshot:
    enabled: bool = True
    backend: str = "disabled"
    master_volume: float = 1.0
    active_layers: List[str] = field(default_factory=list)
    queued_events: int = 0
    pending_cues: List[str] = field(default_factory=list)
    current_mood: str = "idle"
    current_state: str = "IDLE"
    current_progress: float = 0.0
    sync_lag_ms: float = 0.0
    last_event: str = ""

    def debug_lines(self) -> List[str]:
        lines = [
            f"Audio backend: {self.backend} | master {self.master_volume:.2f}",
            f"Audio state: {self.current_state} | mood {self.current_mood}",
            f"Progress: {self.current_progress:.1f}% | sync lag {self.sync_lag_ms:.1f} ms",
            f"Queued cues: {self.queued_events} | active layers: {', '.join(self.active_layers) if self.active_layers else 'none'}",
        ]
        if self.last_event:
            lines.append(f"Last event: {self.last_event}")
        if self.pending_cues:
            lines.append(f"Pending: {', '.join(self.pending_cues)}")
        return lines


def _current_time_ms() -> float:
    return time.perf_counter() * 1000.0
