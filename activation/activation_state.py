"""Activation state definitions."""

from enum import Enum


class ActivationState(Enum):
    """Centralized activation states."""

    IDLE = "IDLE"
    FACE_DETECTED = "FACE_DETECTED"
    TRACKING = "TRACKING"
    STARE_DETECTED = "STARE_DETECTED"
    ACTIVATING = "ACTIVATING"
    POWER_ACTIVE = "POWER_ACTIVE"
    INTERRUPTED = "INTERRUPTED"
    COOLDOWN = "COOLDOWN"
