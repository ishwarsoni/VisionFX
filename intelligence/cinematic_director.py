"""Cinematic Director — top-level facade orchestrating all Phase 8 intelligence subsystems."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from config.settings import IntelligenceConfig
from intelligence.behavior_tracker import BehaviorTracker
from intelligence.escalation_system import EscalationSystem
from intelligence.immersion_controller import ImmersionController
from intelligence.intelligence_data import (
    DirectorSnapshot,
    EscalationTier,
)
from intelligence.memory_system import MemorySystem
from intelligence.personality_engine import PersonalityEngine
from intelligence.rare_event_manager import RareEventManager
from utils.logger import Logger


class CinematicDirector:
    """
    Central coordinator for the adaptive cinematic intelligence layer.

    Follows the same facade pattern as ``EffectManager``, ``AudioManager``,
    and ``RecordingManager``:

    - Owns all Phase 8 subsystems.
    - Receives tracking + activation data from the main loop via ``update()``.
    - Outputs a ``DirectorSnapshot`` that the main loop can pass downstream.
    - Binds to the activation engine via ``bind_activation_engine()``.
    - Has **zero** imports from tracking, effects, or presentation.

    The director's role is cinematic *pacing* — it decides when to be
    subtle and when to explode visually by controlling tension, escalation,
    personality, and rare event scheduling.
    """

    def __init__(
        self,
        config: IntelligenceConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or Logger(name="CinematicDirector")

        # Subsystems
        self.behavior = BehaviorTracker(
            window_s=config.behavior_window_s,
            logger=self.logger,
        )

        self.memory = MemorySystem(
            persist_enabled=config.memory_persistence,
            logger=self.logger,
        )

        self.personality = PersonalityEngine(
            initial_mode=config.initial_personality,
            auto_adapt=config.auto_adapt,
            logger=self.logger,
        )

        self.escalation = EscalationSystem(
            rise_rate=config.escalation_rise_rate,
            decay_rate=config.escalation_decay_rate,
            logger=self.logger,
        )

        self.rare_events = RareEventManager(
            probability_scale=config.rare_event_probability,
            cooldown_s=config.rare_event_cooldown_s,
            logger=self.logger,
        )

        self.immersion = ImmersionController(
            intensity=config.immersion_intensity,
            logger=self.logger,
        )

        # Timing
        self._last_update_time: Optional[float] = None

        # Output
        self.snapshot = DirectorSnapshot()

        self.logger.info("CinematicDirector initialized successfully")

    # ------------------------------------------------------------------
    # Activation engine binding (event-only)
    # ------------------------------------------------------------------

    def bind_activation_engine(self, activation_engine) -> None:
        """Subscribe to activation events for memory and rare-event hooks."""
        if activation_engine is None:
            return

        activation_engine.on(
            "on_activation_complete",
            lambda p: self._on_activation_complete(p),
        )
        activation_engine.on(
            "on_interrupt",
            lambda p: self._on_interrupt(p),
        )
        self.logger.info("CinematicDirector bound to activation engine")

    def _on_activation_complete(self, payload: dict) -> None:
        snap = payload.get("snapshot", {})
        elapsed = snap.get("activation_elapsed_ms", 0.0)
        self.memory.record_activation(elapsed)

    def _on_interrupt(self, payload: dict) -> None:
        # Memory already tracks via update(); this is for future hooks.
        pass

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, tracking_data, activation_data) -> DirectorSnapshot:
        """
        Run all intelligence subsystems for the current frame.

        Args:
            tracking_data: TrackingFrameData (or None).
            activation_data: ActivationSnapshot (or None).

        Returns:
            ``DirectorSnapshot`` with all intelligence outputs.
        """
        now = time.perf_counter()
        delta_s = (
            0.0
            if self._last_update_time is None
            else max(0.0, now - self._last_update_time)
        )
        self._last_update_time = now

        # 1. Behavior tracking
        metrics = self.behavior.update(tracking_data, activation_data)

        # 2. Memory integration
        self.memory.update(metrics, activation_data)

        # 3. Personality adaptation
        personality = self.personality.profile
        if self.config.auto_adapt:
            adapted = self.personality.auto_adapt(metrics, self.memory.memory, delta_s)
            if adapted:
                personality = adapted
                self.memory.record_personality(personality.mode.value)

        # 4. Escalation
        esc_state = self.escalation.update(metrics, personality, delta_s)
        self.memory.record_escalation_peak(esc_state.peak_intensity)

        # 5. Rare events
        rare = self.rare_events.update(
            esc_state,
            metrics,
            self.memory.memory,
            delta_s,
        )
        if rare and rare.active and rare.elapsed_ms <= delta_s * 1000 + 1:
            # Just triggered
            self.memory.record_rare_event()

        # 6. Immersion
        imm = self.immersion.update(esc_state, personality, rare, delta_s)

        # 7. Compute director-level tension and pacing
        tension = self._compute_tension(esc_state, metrics, personality)
        pacing = self._compute_pacing(esc_state, tension)
        suppress = self._should_suppress(esc_state, tension, delta_s)

        # 8. Build output
        self.snapshot = DirectorSnapshot(
            personality=personality,
            escalation=esc_state,
            behavior=metrics,
            immersion=imm,
            rare_event=rare,
            tension=tension,
            pacing_multiplier=pacing,
            suppress_effects=suppress,
        )

        return self.snapshot

    # ------------------------------------------------------------------
    # Tension & pacing
    # ------------------------------------------------------------------

    def _compute_tension(
        self,
        esc: object,
        metrics: object,
        personality: object,
    ) -> float:
        """0.0–1.0 overall dramatic tension."""
        base = personality.baseline_tension
        esc_contrib = esc.intensity * 0.5
        stare_contrib = min(1.0, metrics.avg_stare_duration_ms / 8000.0) * 0.2
        blink_contrib = min(1.0, metrics.no_blink_streak_ms / 10000.0) * 0.15
        return min(1.0, base + esc_contrib + stare_contrib + blink_contrib)

    @staticmethod
    def _compute_pacing(esc, tension: float) -> float:
        """Pacing multiplier: < 1.0 = slow/dramatic, > 1.0 = fast/urgent."""
        if tension < 0.2:
            return 0.7  # slow, ambient
        elif tension < 0.5:
            return 0.9  # building
        elif tension < 0.8:
            return 1.1  # active
        else:
            return 1.3  # peak urgency

    def _should_suppress(self, esc, tension: float, delta_s: float) -> bool:
        """
        Director decides to 'go quiet' — pause effects for dramatic contrast.
        This creates the 'calm before the storm' effect.
        """
        # Suppress briefly when escalation just dropped from peak to building
        if (
            esc.tier == EscalationTier.BUILDING
            and esc.momentum < -0.05
            and tension < 0.3
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Manual controls (for creator mode / hotkeys)
    # ------------------------------------------------------------------

    def set_personality(self, mode: str) -> None:
        """Manually set personality mode."""
        self.personality.set_mode(mode)
        self.memory.record_personality(mode)

    def cycle_personality(self) -> None:
        """Cycle to the next personality."""
        profile = self.personality.cycle_next()
        self.memory.record_personality(profile.mode.value)

    def force_rare_event(self, event_type: str = "ANOMALY_BURST") -> None:
        """Manually trigger a rare event."""
        self.rare_events.force_trigger(event_type)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def get_debug_lines(self) -> List[str]:
        if not self.config.enable_debug:
            return []

        lines = ["── INTELLIGENCE ──"]
        lines.extend(self.snapshot.debug_lines())
        return lines

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Save memory and release resources."""
        self.memory.release()
        self.logger.info("CinematicDirector released")
