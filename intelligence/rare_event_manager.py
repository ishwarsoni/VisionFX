"""Rare event manager — generates unexpected cinematic moments with weighted probability."""

from __future__ import annotations

import random
import time
from typing import List, Optional

from intelligence.intelligence_data import (
    BehaviorMetrics,
    EscalationState,
    EscalationTier,
    RareEvent,
    RareEventType,
    SessionMemory,
)
from utils.logger import Logger

# Base probability per second for each event type (tuned to feel rare)
_BASE_PROBABILITY: dict = {
    RareEventType.ANOMALY_BURST: 0.003,
    RareEventType.VOID_FLASH: 0.002,
    RareEventType.CORRUPTION_SURGE: 0.0025,
    RareEventType.REALITY_COLLAPSE: 0.001,
    RareEventType.FORBIDDEN_MODE: 0.0005,
}

# Duration range (ms) for each event
_DURATION_RANGE: dict = {
    RareEventType.ANOMALY_BURST: (400, 900),
    RareEventType.VOID_FLASH: (200, 600),
    RareEventType.CORRUPTION_SURGE: (600, 1200),
    RareEventType.REALITY_COLLAPSE: (800, 1800),
    RareEventType.FORBIDDEN_MODE: (1200, 2500),
}


class RareEventManager:
    """
    Stochastically triggers rare cinematic moments based on escalation
    level, behavior, and session history.  Higher escalation and longer
    sessions increase the probability of rare events — but they remain
    genuinely rare so each occurrence feels special.

    Only one rare event can be active at a time.
    """

    def __init__(
        self,
        probability_scale: float = 1.0,
        cooldown_s: float = 20.0,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="RareEventManager")
        self._probability_scale = max(0.0, probability_scale)
        self._cooldown_s = max(5.0, cooldown_s)
        self._last_event_time: float = 0.0

        self.active_event: Optional[RareEvent] = None

        self.logger.info(
            f"RareEventManager initialized (scale={self._probability_scale:.2f})"
        )

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        escalation: EscalationState,
        behavior: BehaviorMetrics,
        memory: SessionMemory,
        delta_s: float,
    ) -> Optional[RareEvent]:
        """
        Advance the active event (if any) or roll for a new one.

        Returns:
            The active ``RareEvent`` if one is running, else ``None``.
        """
        # Advance active event
        if self.active_event and self.active_event.active:
            self.active_event.elapsed_ms += delta_s * 1000.0
            if self.active_event.is_complete:
                self.logger.info(
                    f"Rare event ended: {self.active_event.event_type.value}"
                )
                self.active_event.active = False
                self.active_event = None
            return self.active_event

        # Cooldown between events
        now = time.perf_counter()
        if now - self._last_event_time < self._cooldown_s:
            return None

        # Only trigger during elevated escalation
        if escalation.tier in (EscalationTier.DORMANT, EscalationTier.AWAKENING):
            return None

        # Roll for each event type
        modifier = self._compute_modifier(escalation, behavior, memory)

        for event_type, base_prob in _BASE_PROBABILITY.items():
            adjusted_prob = base_prob * self._probability_scale * modifier * delta_s
            if random.random() < adjusted_prob:
                self._trigger(event_type, escalation.intensity)
                return self.active_event

        return None

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def _trigger(self, event_type: RareEventType, intensity: float) -> None:
        dur_min, dur_max = _DURATION_RANGE.get(event_type, (500, 1000))
        duration = random.uniform(dur_min, dur_max)

        self.active_event = RareEvent(
            event_type=event_type,
            intensity=min(1.0, intensity * random.uniform(0.8, 1.2)),
            duration_ms=duration,
            elapsed_ms=0.0,
            active=True,
        )
        self._last_event_time = time.perf_counter()

        self.logger.info(
            f"RARE EVENT: {event_type.value} "
            f"(intensity={self.active_event.intensity:.2f}, "
            f"duration={duration:.0f}ms)"
        )

    # ------------------------------------------------------------------
    # Probability modifier
    # ------------------------------------------------------------------

    def _compute_modifier(
        self,
        escalation: EscalationState,
        behavior: BehaviorMetrics,
        memory: SessionMemory,
    ) -> float:
        """
        Compute a multiplier that makes rare events more likely when the
        session is deeply engaged.
        """
        mod = 1.0

        # Escalation intensity is the primary driver
        mod *= 1.0 + escalation.intensity * 3.0

        # Long sessions increase probability slightly
        if behavior.session_duration_s > 120:
            mod *= 1.3
        if behavior.session_duration_s > 300:
            mod *= 1.5

        # Fewer rare events already triggered → higher chance
        if memory.rare_events_triggered == 0:
            mod *= 1.5
        elif memory.rare_events_triggered < 3:
            mod *= 1.2

        # No-blink streak boosts rare event chance
        if behavior.no_blink_streak_ms > 5000:
            mod *= 1.4

        return mod

    # ------------------------------------------------------------------
    # Manual trigger (for creator mode)
    # ------------------------------------------------------------------

    def force_trigger(
        self, event_type_str: str = "ANOMALY_BURST"
    ) -> Optional[RareEvent]:
        """Manually trigger a rare event (for testing / creator mode)."""
        try:
            event_type = RareEventType(event_type_str.upper())
        except ValueError:
            event_type = RareEventType.ANOMALY_BURST

        self._trigger(event_type, 1.0)
        return self.active_event

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.active_event is not None and self.active_event.active

    def get_debug_lines(self) -> List[str]:
        lines = []
        if self.active_event and self.active_event.active:
            e = self.active_event
            lines.append(f"RARE: {e.event_type.value} ({e.progress:.0%})")
        else:
            cd = max(
                0, self._cooldown_s - (time.perf_counter() - self._last_event_time)
            )
            lines.append(f"RARE: IDLE (cd {cd:.0f}s)")
        return lines
