"""Shared data structures for the visual effects engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from activation.activation_data import ActivationSnapshot
from activation.activation_state import ActivationState
from tracking.tracking_data import TrackingFrameData


@dataclass
class EffectContext:
    """Frame-level rendering context for effects."""

    frame_index: int = 0
    timestamp_ms: float = 0.0
    delta_ms: float = 0.0
    frame_size: Tuple[int, int] = (0, 0)
    quality: int = 2
    debug_enabled: bool = False
    tracking_data: Optional[TrackingFrameData] = None
    activation_data: Optional[ActivationSnapshot] = None
    activation_state: ActivationState = ActivationState.IDLE
    activation_progress: float = 0.0
    activation_ready: bool = False
    events: Dict[str, Any] = field(default_factory=dict)
    raw_frame: Any = None


@dataclass
class EffectDebugState:
    """Minimal debug telemetry for the compositor."""

    active_effects: List[str] = field(default_factory=list)
    compositor_order: List[str] = field(default_factory=list)
    effect_intensities: Dict[str, float] = field(default_factory=dict)
    render_time_ms: float = 0.0
    frame_index: int = 0

    def lines(self) -> List[str]:
        lines = [f"FX RENDER: {self.render_time_ms:.2f}ms"]
        if self.compositor_order:
            lines.append(f"FX ORDER: {' > '.join(self.compositor_order)}")
        if self.active_effects:
            lines.append(f"FX ACTIVE: {', '.join(self.active_effects)}")
        if self.effect_intensities:
            compact = ", ".join(
                f"{name}:{value:.2f}" for name, value in self.effect_intensities.items()
            )
            lines.append(f"FX INTENSITY: {compact}")
        return lines
