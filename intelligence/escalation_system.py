"""Adaptive effect escalation system driven by behavioral metrics."""

from __future__ import annotations

import math
from typing import Optional

from intelligence.intelligence_data import (
    BehaviorMetrics,
    EscalationState,
    EscalationTier,
    PersonalityProfile,
)
from utils.logger import Logger

# Tier thresholds on the 0.0–1.0 intensity scale
_TIER_THRESHOLDS = {
    EscalationTier.DORMANT: 0.0,
    EscalationTier.AWAKENING: 0.15,
    EscalationTier.BUILDING: 0.40,
    EscalationTier.PEAK: 0.70,
    EscalationTier.OVERLOAD: 0.90,
}


class EscalationSystem:
    """
    Manages a continuous 0.0–1.0 escalation intensity that rises with
    sustained engagement and decays during inactivity.  The escalation
    state is consumed by the director to modulate all downstream
    cinematic effects.

    Escalation is intentionally inertial — it builds slowly and decays
    slowly so the experience feels dramatic rather than twitchy.
    """

    def __init__(
        self,
        rise_rate: float = 0.08,
        decay_rate: float = 0.03,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="EscalationSystem")
        self._rise_rate = max(0.01, rise_rate)
        self._decay_rate = max(0.005, decay_rate)

        self.state = EscalationState()

        self.logger.info("EscalationSystem initialized")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        behavior: BehaviorMetrics,
        personality: PersonalityProfile,
        delta_s: float,
    ) -> EscalationState:
        """
        Advance escalation intensity based on current behavior.

        Args:
            behavior: Current behavioral metrics.
            personality: Active personality profile.
            delta_s: Seconds since last frame.

        Returns:
            Updated ``EscalationState``.
        """
        if delta_s <= 0:
            return self.state

        # Compute engagement signal (0.0–1.0)
        engagement = self._compute_engagement(behavior)

        # Apply personality modifiers
        speed = personality.activation_speed

        # Rise when engaged, decay when not
        if engagement > 0.3:
            drive = engagement * self._rise_rate * speed * delta_s
            self.state.intensity = min(1.0, self.state.intensity + drive)
            self.state.momentum = drive / max(0.001, delta_s)
        else:
            decay = self._decay_rate * delta_s
            self.state.intensity = max(0.0, self.state.intensity - decay)
            self.state.momentum = -decay / max(0.001, delta_s)

        # Track peak
        if self.state.intensity > self.state.peak_intensity:
            self.state.peak_intensity = self.state.intensity

        # Count peak events
        if self.state.intensity >= 0.9 and self.state.tier != EscalationTier.OVERLOAD:
            self.state.escalation_count += 1

        # Determine tier
        self.state.tier = self._resolve_tier(self.state.intensity)

        return self.state

    # ------------------------------------------------------------------
    # Engagement scoring
    # ------------------------------------------------------------------

    def _compute_engagement(self, b: BehaviorMetrics) -> float:
        """
        Derive a 0.0–1.0 engagement score from behavior metrics.
        Higher engagement → faster escalation.
        """
        score = 0.0

        # Stare duration contributes heavily
        if b.avg_stare_duration_ms > 500:
            stare_factor = min(1.0, b.avg_stare_duration_ms / 5000.0)
            score += stare_factor * 0.35

        # No-blink streak is a strong engagement signal
        if b.no_blink_streak_ms > 2000:
            no_blink_factor = min(1.0, b.no_blink_streak_ms / 15000.0)
            score += no_blink_factor * 0.25

        # Activation frequency
        if b.activation_rate_per_min > 0:
            act_factor = min(1.0, b.activation_rate_per_min / 6.0)
            score += act_factor * 0.2

        # Low interruption rate = focused user
        if b.activation_count > 0:
            focus = 1.0 - b.interruption_rate
            score += focus * 0.1

        # Low movement = intense focus
        if b.movement_energy < 0.01:
            score += 0.1

        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Tier resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tier(intensity: float) -> EscalationTier:
        result = EscalationTier.DORMANT
        for tier, threshold in _TIER_THRESHOLDS.items():
            if intensity >= threshold:
                result = tier
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.state = EscalationState()

    def get_debug_lines(self) -> list:
        s = self.state
        return [
            f"ESCAL: {s.tier.value} ({s.intensity:.2f})",
            f"MOMENTUM: {s.momentum:+.3f}",
            f"PEAK: {s.peak_intensity:.2f} x{s.escalation_count}",
        ]
