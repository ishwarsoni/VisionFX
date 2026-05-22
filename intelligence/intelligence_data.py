"""Data containers for Phase 8 adaptive intelligence, personality, and cinematic direction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Personality profiles
# ---------------------------------------------------------------------------


class PersonalityMode(Enum):
    """Cinematic personality archetypes."""

    CALM_OBSERVER = "CALM_OBSERVER"
    CORRUPTED_ENTITY = "CORRUPTED_ENTITY"
    UNSTABLE_POWER = "UNSTABLE_POWER"
    AGGRESSIVE_AWAKENING = "AGGRESSIVE_AWAKENING"
    SILENT_VOID = "SILENT_VOID"


@dataclass
class PersonalityProfile:
    """Resolved parameters for a personality mode."""

    mode: PersonalityMode = PersonalityMode.CALM_OBSERVER
    label: str = "Calm Observer"

    # Effect modifiers (multiplied onto existing effect intensities)
    glitch_scale: float = 1.0
    distortion_scale: float = 1.0
    bloom_scale: float = 1.0
    impact_scale: float = 1.0
    screen_fx_scale: float = 1.0

    # Presentation modifiers
    hud_scale: float = 1.0
    corruption_scale: float = 1.0
    immersion_scale: float = 1.0
    zoom_scale: float = 1.0

    # Timing modifiers (multiplied onto durations)
    activation_speed: float = 1.0
    cooldown_speed: float = 1.0
    transition_speed: float = 1.0

    # Color temperature shift (-1.0 cold … +1.0 warm)
    color_temperature: float = 0.0

    # Pacing tension baseline (0.0 relaxed … 1.0 hostile)
    baseline_tension: float = 0.0


# ---------------------------------------------------------------------------
# Escalation state
# ---------------------------------------------------------------------------


class EscalationTier(Enum):
    """Cinematic escalation tiers."""

    DORMANT = "DORMANT"
    AWAKENING = "AWAKENING"
    BUILDING = "BUILDING"
    PEAK = "PEAK"
    OVERLOAD = "OVERLOAD"


@dataclass
class EscalationState:
    """Current escalation level and intensity."""

    tier: EscalationTier = EscalationTier.DORMANT
    intensity: float = 0.0  # 0.0 – 1.0 continuous
    momentum: float = 0.0  # rate of change
    peak_intensity: float = 0.0  # highest reached this session
    escalation_count: int = 0  # times peak was reached


# ---------------------------------------------------------------------------
# Behavior metrics
# ---------------------------------------------------------------------------


@dataclass
class BehaviorMetrics:
    """Rolling behavioral statistics derived from tracking + activation."""

    # Stare behaviour
    avg_stare_duration_ms: float = 0.0
    max_stare_duration_ms: float = 0.0
    stare_count: int = 0

    # Blink behaviour
    blink_rate_per_min: float = 0.0
    no_blink_streak_ms: float = 0.0

    # Activation behaviour
    activation_count: int = 0
    activation_rate_per_min: float = 0.0
    avg_activation_duration_ms: float = 0.0
    interruption_count: int = 0
    interruption_rate: float = 0.0

    # Movement / presence
    face_absence_count: int = 0
    face_absence_total_ms: float = 0.0
    movement_energy: float = 0.0

    # Session timing
    session_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Memory record
# ---------------------------------------------------------------------------


@dataclass
class SessionMemory:
    """Lightweight session memory for cinematic continuity."""

    total_activations: int = 0
    longest_stare_ms: float = 0.0
    peak_escalation: float = 0.0
    dominant_personality: str = "CALM_OBSERVER"
    total_session_time_s: float = 0.0
    rare_events_triggered: int = 0
    interruption_patterns: Dict[str, int] = field(default_factory=dict)
    activation_history: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_activations": self.total_activations,
            "longest_stare_ms": round(self.longest_stare_ms, 1),
            "peak_escalation": round(self.peak_escalation, 3),
            "dominant_personality": self.dominant_personality,
            "total_session_time_s": round(self.total_session_time_s, 1),
            "rare_events_triggered": self.rare_events_triggered,
            "interruption_patterns": dict(self.interruption_patterns),
        }


# ---------------------------------------------------------------------------
# Rare events
# ---------------------------------------------------------------------------


class RareEventType(Enum):
    """Types of rare cinematic moments."""

    REALITY_COLLAPSE = "REALITY_COLLAPSE"
    FORBIDDEN_MODE = "FORBIDDEN_MODE"
    ANOMALY_BURST = "ANOMALY_BURST"
    VOID_FLASH = "VOID_FLASH"
    CORRUPTION_SURGE = "CORRUPTION_SURGE"


@dataclass
class RareEvent:
    """A triggered rare cinematic event."""

    event_type: RareEventType = RareEventType.ANOMALY_BURST
    intensity: float = 1.0
    duration_ms: float = 800.0
    elapsed_ms: float = 0.0
    active: bool = False

    @property
    def progress(self) -> float:
        if self.duration_ms <= 0:
            return 1.0
        return min(1.0, self.elapsed_ms / self.duration_ms)

    @property
    def is_complete(self) -> bool:
        return self.elapsed_ms >= self.duration_ms


# ---------------------------------------------------------------------------
# Immersion state
# ---------------------------------------------------------------------------


@dataclass
class ImmersionState:
    """Current immersion modifiers computed by the immersion controller."""

    tunnel_vision: float = 0.0  # 0.0 – 1.0
    color_shift: float = 0.0  # -1.0 cold … +1.0 warm
    breathing_phase: float = 0.0  # 0.0 – 1.0 cyclic
    heartbeat_phase: float = 0.0  # 0.0 – 1.0 cyclic
    instability: float = 0.0  # 0.0 – 1.0 environmental shake
    pressure: float = 0.0  # 0.0 – 1.0 overall immersive intensity


# ---------------------------------------------------------------------------
# Director snapshot (output of the intelligence layer each frame)
# ---------------------------------------------------------------------------


@dataclass
class DirectorSnapshot:
    """Frame-level intelligence output consumed by the main loop."""

    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    escalation: EscalationState = field(default_factory=EscalationState)
    behavior: BehaviorMetrics = field(default_factory=BehaviorMetrics)
    immersion: ImmersionState = field(default_factory=ImmersionState)
    rare_event: Optional[RareEvent] = None
    tension: float = 0.0  # 0.0 – 1.0 overall dramatic tension
    pacing_multiplier: float = 1.0  # speed adjustment for cinematic pacing
    suppress_effects: bool = False  # director says "stay quiet"

    def debug_lines(self) -> List[str]:
        lines = [
            f"PERSONA: {self.personality.mode.value}",
            f"ESCAL: {self.escalation.tier.value} ({self.escalation.intensity:.2f})",
            f"TENSION: {self.tension:.2f} | PACE: {self.pacing_multiplier:.2f}",
            f"IMMERSION: P{self.immersion.pressure:.2f} T{self.immersion.tunnel_vision:.2f}",
        ]
        if self.rare_event and self.rare_event.active:
            lines.append(
                f"RARE: {self.rare_event.event_type.value} ({self.rare_event.progress:.0%})"
            )
        if self.suppress_effects:
            lines.append("DIRECTOR: SUPPRESS")
        return lines
