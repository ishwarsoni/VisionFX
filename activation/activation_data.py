"""Data containers for activation state output and debug information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from activation.activation_state import ActivationState


@dataclass
class ActivationSnapshot:
    """Frame-level activation summary."""

    state: ActivationState = ActivationState.IDLE
    activation_progress: float = 0.0
    activation_ready: bool = False
    stare_duration_ms: float = 0.0
    activation_elapsed_ms: float = 0.0
    cooldown_remaining_ms: float = 0.0
    interruption_duration_ms: float = 0.0
    tracking_confidence: float = 0.0
    face_detected: bool = False
    eye_contact: bool = False
    blink_detected: bool = False
    interruption_reason: str = ""
    transition_reason: str = ""
    activation_count: int = 0
    details: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.value,
            "activation_progress": round(self.activation_progress, 2),
            "activation_ready": self.activation_ready,
            "stare_duration_ms": round(self.stare_duration_ms, 2),
            "activation_elapsed_ms": round(self.activation_elapsed_ms, 2),
            "cooldown_remaining_ms": round(self.cooldown_remaining_ms, 2),
            "interruption_duration_ms": round(self.interruption_duration_ms, 2),
            "tracking_confidence": round(self.tracking_confidence, 3),
            "face_detected": self.face_detected,
            "eye_contact": self.eye_contact,
            "blink_detected": self.blink_detected,
            "interruption_reason": self.interruption_reason,
            "transition_reason": self.transition_reason,
            "activation_count": self.activation_count,
            "details": dict(self.details),
        }

    def debug_lines(self) -> List[str]:
        return [
            f"ACT: {self.state.value}",
            f"PROG: {self.activation_progress:.0f}% | READY: {'YES' if self.activation_ready else 'NO'}",
            f"STARE: {self.stare_duration_ms / 1000.0:.2f}s",
            f"ACT TIMER: {self.activation_elapsed_ms / 1000.0:.2f}s",
            f"COOLDOWN: {self.cooldown_remaining_ms / 1000.0:.2f}s",
            f"CONF: {self.tracking_confidence:.2f}",
            f"INT: {self.interruption_reason or 'none'}",
        ]
