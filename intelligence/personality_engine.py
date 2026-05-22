"""Personality engine with cinematic archetype profiles and adaptive switching."""

from __future__ import annotations

from typing import Dict, List, Optional

from intelligence.intelligence_data import (
    BehaviorMetrics,
    PersonalityMode,
    PersonalityProfile,
    SessionMemory,
)
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Predefined personality profiles
# ---------------------------------------------------------------------------

_PROFILES: Dict[PersonalityMode, PersonalityProfile] = {
    PersonalityMode.CALM_OBSERVER: PersonalityProfile(
        mode=PersonalityMode.CALM_OBSERVER,
        label="Calm Observer",
        glitch_scale=0.4,
        distortion_scale=0.3,
        bloom_scale=1.2,
        impact_scale=0.6,
        screen_fx_scale=0.8,
        hud_scale=0.7,
        corruption_scale=0.2,
        immersion_scale=0.5,
        zoom_scale=0.5,
        activation_speed=0.8,
        cooldown_speed=1.2,
        transition_speed=0.7,
        color_temperature=-0.15,
        baseline_tension=0.1,
    ),
    PersonalityMode.CORRUPTED_ENTITY: PersonalityProfile(
        mode=PersonalityMode.CORRUPTED_ENTITY,
        label="Corrupted Entity",
        glitch_scale=1.6,
        distortion_scale=1.4,
        bloom_scale=0.6,
        impact_scale=1.3,
        screen_fx_scale=1.5,
        hud_scale=1.2,
        corruption_scale=1.8,
        immersion_scale=1.3,
        zoom_scale=1.1,
        activation_speed=1.2,
        cooldown_speed=0.7,
        transition_speed=1.4,
        color_temperature=0.3,
        baseline_tension=0.6,
    ),
    PersonalityMode.UNSTABLE_POWER: PersonalityProfile(
        mode=PersonalityMode.UNSTABLE_POWER,
        label="Unstable Power",
        glitch_scale=1.3,
        distortion_scale=1.8,
        bloom_scale=1.0,
        impact_scale=1.5,
        screen_fx_scale=1.2,
        hud_scale=0.9,
        corruption_scale=1.4,
        immersion_scale=1.1,
        zoom_scale=1.4,
        activation_speed=1.4,
        cooldown_speed=0.5,
        transition_speed=1.6,
        color_temperature=0.1,
        baseline_tension=0.45,
    ),
    PersonalityMode.AGGRESSIVE_AWAKENING: PersonalityProfile(
        mode=PersonalityMode.AGGRESSIVE_AWAKENING,
        label="Aggressive Awakening",
        glitch_scale=1.1,
        distortion_scale=1.2,
        bloom_scale=1.5,
        impact_scale=1.8,
        screen_fx_scale=1.4,
        hud_scale=1.3,
        corruption_scale=1.0,
        immersion_scale=1.5,
        zoom_scale=1.6,
        activation_speed=1.6,
        cooldown_speed=0.6,
        transition_speed=1.3,
        color_temperature=0.45,
        baseline_tension=0.7,
    ),
    PersonalityMode.SILENT_VOID: PersonalityProfile(
        mode=PersonalityMode.SILENT_VOID,
        label="Silent Void",
        glitch_scale=0.2,
        distortion_scale=0.8,
        bloom_scale=0.4,
        impact_scale=0.3,
        screen_fx_scale=1.3,
        hud_scale=0.3,
        corruption_scale=0.5,
        immersion_scale=1.6,
        zoom_scale=0.7,
        activation_speed=0.6,
        cooldown_speed=1.5,
        transition_speed=0.5,
        color_temperature=-0.4,
        baseline_tension=0.3,
    ),
}


class PersonalityEngine:
    """
    Manages cinematic personality archetypes and supports both manual
    switching and behavior-driven automatic adaptation.

    Each personality is a set of multipliers that the director layer
    applies onto existing effect/presentation parameters — the
    personality engine never mutates external configs directly.
    """

    def __init__(
        self,
        initial_mode: str = "CALM_OBSERVER",
        auto_adapt: bool = True,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="PersonalityEngine")
        self._auto_adapt = auto_adapt
        self._mode = self._parse_mode(initial_mode)
        self._profile = _PROFILES[self._mode]
        self._adapt_cooldown: float = 0.0  # seconds remaining before next auto-switch

        self.logger.info(f"PersonalityEngine initialized: {self._mode.value}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> PersonalityMode:
        return self._mode

    @property
    def profile(self) -> PersonalityProfile:
        return self._profile

    # ------------------------------------------------------------------
    # Manual switching
    # ------------------------------------------------------------------

    def set_mode(self, mode_str: str) -> PersonalityProfile:
        """Switch personality by name."""
        new_mode = self._parse_mode(mode_str)
        if new_mode != self._mode:
            self._mode = new_mode
            self._profile = _PROFILES[self._mode]
            self._adapt_cooldown = 30.0  # prevent rapid re-adaptation
            self.logger.info(f"Personality changed to {self._mode.value}")
        return self._profile

    def cycle_next(self) -> PersonalityProfile:
        """Cycle to the next personality in order."""
        modes = list(PersonalityMode)
        idx = (modes.index(self._mode) + 1) % len(modes)
        return self.set_mode(modes[idx].value)

    # ------------------------------------------------------------------
    # Automatic adaptation
    # ------------------------------------------------------------------

    def auto_adapt(
        self,
        behavior: BehaviorMetrics,
        memory: SessionMemory,
        delta_s: float,
    ) -> Optional[PersonalityProfile]:
        """
        Recommend a personality switch based on behavioral patterns.
        Returns the new profile only if a switch is recommended.
        """
        if not self._auto_adapt:
            return None

        # Cooldown between auto-switches
        if self._adapt_cooldown > 0:
            self._adapt_cooldown -= delta_s
            return None

        suggested = self._suggest_personality(behavior, memory)
        if suggested != self._mode:
            self.logger.info(
                f"Auto-adapt: {self._mode.value} -> {suggested.value} "
                f"(stare={behavior.avg_stare_duration_ms:.0f}ms, "
                f"acts={behavior.activation_count}, "
                f"blinks={behavior.blink_rate_per_min:.0f}/min)"
            )
            self._mode = suggested
            self._profile = _PROFILES[self._mode]
            self._adapt_cooldown = 45.0  # cooldown before next adaptation
            return self._profile

        return None

    def _suggest_personality(
        self, b: BehaviorMetrics, m: SessionMemory
    ) -> PersonalityMode:
        """Score-based personality recommendation."""
        # Very long stares + low blink rate -> Silent Void or Aggressive
        if b.no_blink_streak_ms > 8000 and b.avg_stare_duration_ms > 3000:
            return PersonalityMode.AGGRESSIVE_AWAKENING

        # High activation rate + many interruptions -> Unstable Power
        if b.activation_rate_per_min > 3.0 and b.interruption_rate > 0.4:
            return PersonalityMode.UNSTABLE_POWER

        # High activation count + low interruption -> Corrupted Entity
        if m.total_activations > 5 and b.interruption_rate < 0.2:
            return PersonalityMode.CORRUPTED_ENTITY

        # Low movement + low blink rate -> Silent Void
        if b.movement_energy < 0.005 and b.blink_rate_per_min < 8:
            return PersonalityMode.SILENT_VOID

        # High movement energy -> Aggressive
        if b.movement_energy > 0.03:
            return PersonalityMode.AGGRESSIVE_AWAKENING

        # Default
        return PersonalityMode.CALM_OBSERVER

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mode(value: str) -> PersonalityMode:
        try:
            return PersonalityMode(value.upper())
        except (ValueError, AttributeError):
            return PersonalityMode.CALM_OBSERVER

    @staticmethod
    def available_modes() -> List[str]:
        return [m.value for m in PersonalityMode]

    def get_debug_lines(self) -> List[str]:
        p = self._profile
        return [
            f"PERSONA: {p.label}",
            f"GLITCH: x{p.glitch_scale:.1f} | DIST: x{p.distortion_scale:.1f}",
            f"IMPACT: x{p.impact_scale:.1f} | BLOOM: x{p.bloom_scale:.1f}",
            f"TENSION: {p.baseline_tension:.2f} | TEMP: {p.color_temperature:+.2f}",
            f"ADAPT CD: {max(0, self._adapt_cooldown):.0f}s",
        ]
