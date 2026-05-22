"""Shared data structures for the cinematic presentation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from activation.activation_data import ActivationSnapshot
from activation.activation_state import ActivationState
from tracking.tracking_data import TrackingFrameData


class PresentationStage(Enum):
    """Cinematic presentation stages."""

    IDLE = "IDLE"
    EYE_LOCK = "EYE_LOCK"
    BUILDUP = "BUILDUP"
    CORRUPTION = "CORRUPTION"
    TRANSFORM = "TRANSFORM"
    INTERRUPTED = "INTERRUPTED"
    COOLDOWN = "COOLDOWN"


@dataclass
class PresentationSnapshot:
    """Frame-level presentation summary."""

    stage: PresentationStage = PresentationStage.IDLE
    stage_progress: float = 0.0
    hud_intensity: float = 0.0
    camera_motion: float = 0.0
    corruption_intensity: float = 0.0
    immersion_intensity: float = 0.0
    text_intensity: float = 0.0
    transition_intensity: float = 0.0
    activation_progress: float = 0.0
    activation_state: ActivationState = ActivationState.IDLE
    active_layers: List[str] = field(default_factory=list)
    queued_transitions: List[str] = field(default_factory=list)
    interruption_reason: str = ""
    debug_label: str = ""

    def lines(self) -> List[str]:
        return [
            f"PRES: {self.stage.value}",
            f"PRES PROG: {self.stage_progress:.0f}%",
            f"HUD: {self.hud_intensity:.2f} | CAM: {self.camera_motion:.2f}",
            f"CORR: {self.corruption_intensity:.2f} | IMM: {self.immersion_intensity:.2f}",
            f"TEXT: {self.text_intensity:.2f} | TRANS: {self.transition_intensity:.2f}",
            f"ACT: {self.activation_state.value} @ {self.activation_progress:.0f}%",
            f"INT: {self.interruption_reason or 'none'}",
        ]


@dataclass
class PresentationContext:
    """Per-frame rendering context for the presentation layer."""

    frame_index: int = 0
    timestamp_ms: float = 0.0
    delta_ms: float = 0.0
    frame_size: Tuple[int, int] = (0, 0)
    activation_data: Optional[ActivationSnapshot] = None
    tracking_data: Optional[TrackingFrameData] = None
    stage: PresentationStage = PresentationStage.IDLE
    stage_progress: float = 0.0
    activation_progress: float = 0.0
    activation_ready: bool = False
    events: Dict[str, Any] = field(default_factory=dict)
