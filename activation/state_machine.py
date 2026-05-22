"""State machine for activation transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from activation.activation_state import ActivationState


@dataclass
class ActivationStateMachine:
    """Validated state machine with transition cooldown protection."""

    transition_cooldown_ms: float = 80.0
    current_state: ActivationState = ActivationState.IDLE
    last_transition_time_ms: float = 0.0
    last_transition_reason: str = ""
    history: List[Tuple[float, ActivationState, ActivationState, str]] = field(
        default_factory=list
    )

    ALLOWED_TRANSITIONS: Dict[ActivationState, Tuple[ActivationState, ...]] = field(
        default_factory=lambda: {
            ActivationState.IDLE: (
                ActivationState.FACE_DETECTED,
                ActivationState.TRACKING,
                ActivationState.COOLDOWN,
            ),
            ActivationState.FACE_DETECTED: (
                ActivationState.TRACKING,
                ActivationState.STARE_DETECTED,
                ActivationState.IDLE,
                ActivationState.INTERRUPTED,
                ActivationState.COOLDOWN,
            ),
            ActivationState.TRACKING: (
                ActivationState.FACE_DETECTED,
                ActivationState.STARE_DETECTED,
                ActivationState.IDLE,
                ActivationState.INTERRUPTED,
                ActivationState.COOLDOWN,
            ),
            ActivationState.STARE_DETECTED: (
                ActivationState.ACTIVATING,
                ActivationState.TRACKING,
                ActivationState.INTERRUPTED,
                ActivationState.COOLDOWN,
            ),
            ActivationState.ACTIVATING: (
                ActivationState.POWER_ACTIVE,
                ActivationState.INTERRUPTED,
                ActivationState.COOLDOWN,
            ),
            ActivationState.POWER_ACTIVE: (
                ActivationState.COOLDOWN,
                ActivationState.INTERRUPTED,
                ActivationState.TRACKING,
                ActivationState.FACE_DETECTED,
                ActivationState.IDLE,
            ),
            ActivationState.INTERRUPTED: (
                ActivationState.COOLDOWN,
                ActivationState.IDLE,
                ActivationState.FACE_DETECTED,
                ActivationState.TRACKING,
            ),
            ActivationState.COOLDOWN: (
                ActivationState.IDLE,
                ActivationState.FACE_DETECTED,
                ActivationState.TRACKING,
            ),
        }
    )

    def can_transition(
        self, target_state: ActivationState, now_ms: float, force: bool = False
    ) -> bool:
        if target_state == self.current_state:
            return True

        if not force and self.last_transition_time_ms > 0.0:
            elapsed_ms = now_ms - self.last_transition_time_ms
            if elapsed_ms < self.transition_cooldown_ms:
                return False

        allowed_targets = self.ALLOWED_TRANSITIONS.get(self.current_state, ())
        return force or target_state in allowed_targets

    def transition(
        self,
        target_state: ActivationState,
        now_ms: float,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        if not self.can_transition(target_state, now_ms, force=force):
            return False

        if target_state == self.current_state:
            self.last_transition_reason = reason
            return True

        previous_state = self.current_state
        self.current_state = target_state
        self.last_transition_time_ms = now_ms
        self.last_transition_reason = reason
        self.history.append((now_ms, previous_state, target_state, reason))
        return True

    def reset(self) -> None:
        self.current_state = ActivationState.IDLE
        self.last_transition_time_ms = 0.0
        self.last_transition_reason = ""
        self.history.clear()
