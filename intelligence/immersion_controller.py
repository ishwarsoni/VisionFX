"""Immersion controller — generates cyclic and adaptive immersion modifiers."""

from __future__ import annotations

import math
from typing import Optional

from intelligence.intelligence_data import (
    EscalationState,
    ImmersionState,
    PersonalityProfile,
    RareEvent,
)
from utils.logger import Logger


class ImmersionController:
    """
    Computes per-frame immersion parameters (tunnel vision, color shift,
    breathing rhythm, heartbeat, instability) from escalation state
    and personality profile.

    The controller generates smooth, organic-feeling modifiers that the
    presentation and effects layers can multiply into their rendering
    without the immersion controller knowing about their internals.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="ImmersionController")
        self._intensity = max(0.0, min(2.0, intensity))
        self._phase_time: float = 0.0

        self.state = ImmersionState()

        self.logger.info(
            f"ImmersionController initialized (intensity={self._intensity:.2f})"
        )

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        escalation: EscalationState,
        personality: PersonalityProfile,
        rare_event: Optional[RareEvent],
        delta_s: float,
    ) -> ImmersionState:
        """
        Compute immersion modifiers for the current frame.

        Args:
            escalation: Current escalation state.
            personality: Active personality profile.
            rare_event: Active rare event (or None).
            delta_s: Seconds since last frame.

        Returns:
            Updated ``ImmersionState``.
        """
        self._phase_time += delta_s

        esc = escalation.intensity
        p = personality

        # ── Tunnel vision: ramps with escalation and personality ──
        tunnel_target = esc * p.immersion_scale * 0.7 * self._intensity
        self.state.tunnel_vision = self._smooth(
            self.state.tunnel_vision, tunnel_target, delta_s, 2.0
        )

        # ── Color shift: personality temperature + escalation warmth ──
        temp_target = p.color_temperature + esc * 0.2
        self.state.color_shift = self._smooth(
            self.state.color_shift, temp_target, delta_s, 1.5
        )

        # ── Breathing rhythm: slow sine wave that speeds up with escalation ──
        breath_speed = 0.3 + esc * 0.5  # 0.3–0.8 Hz
        self.state.breathing_phase = (
            math.sin(self._phase_time * breath_speed * 2 * math.pi) + 1.0
        ) / 2.0

        # ── Heartbeat: faster pulse that intensifies at peak ──
        heart_speed = 0.8 + esc * 1.5  # 0.8–2.3 Hz
        raw_beat = math.sin(self._phase_time * heart_speed * 2 * math.pi)
        # Sharpen to pulse shape
        self.state.heartbeat_phase = max(0.0, raw_beat) ** 3

        # ── Environmental instability: noise-like shimmer ──
        instability_base = esc * p.corruption_scale * 0.3
        noise = math.sin(self._phase_time * 7.3) * math.sin(self._phase_time * 13.7)
        instability_target = instability_base + abs(noise) * esc * 0.2
        self.state.instability = self._smooth(
            self.state.instability, instability_target, delta_s, 4.0
        )

        # ── Rare event surge ──
        if rare_event and rare_event.active:
            surge = rare_event.intensity * (1.0 - rare_event.progress)
            self.state.tunnel_vision = min(1.0, self.state.tunnel_vision + surge * 0.3)
            self.state.instability = min(1.0, self.state.instability + surge * 0.5)

        # ── Overall pressure ──
        self.state.pressure = min(
            1.0,
            (
                self.state.tunnel_vision * 0.3
                + abs(self.state.color_shift) * 0.15
                + self.state.heartbeat_phase * 0.2
                + self.state.instability * 0.2
                + esc * 0.15
            )
            * self._intensity,
        )

        return self.state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(current: float, target: float, delta_s: float, speed: float) -> float:
        """Exponential smoothing toward target."""
        alpha = 1.0 - math.exp(-speed * delta_s) if delta_s > 0 else 0.0
        return current + (target - current) * alpha

    def reset(self) -> None:
        self.state = ImmersionState()
        self._phase_time = 0.0

    def get_debug_lines(self) -> list:
        s = self.state
        return [
            f"IMMERSE: P{s.pressure:.2f} T{s.tunnel_vision:.2f}",
            f"BREATH: {s.breathing_phase:.2f} | BEAT: {s.heartbeat_phase:.2f}",
            f"INSTAB: {s.instability:.2f} | TEMP: {s.color_shift:+.2f}",
        ]
