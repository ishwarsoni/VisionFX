"""Ambient immersion audio planning."""

from __future__ import annotations

import math

import numpy as np

from audio.audio_data import (
    AudioCue,
    AudioEventContext,
    AudioLayer,
    AudioPlan,
    LayerTarget,
)
from audio.audio_transition import clamp


class AmbienceSystem:
    def __init__(
        self,
        ambience_volume: float,
        transition_fade_ms: float,
        spatialization_intensity: float,
    ) -> None:
        self.ambience_volume = ambience_volume
        self.transition_fade_ms = transition_fade_ms
        self.spatialization_intensity = spatialization_intensity

    def on_event(self, event_name: str, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="ambient")
        if event_name in ("on_stare_start", "on_eye_lock", "on_stare_detected"):
            plan.cues.append(self._pressure_breath(context))
        return plan

    def plan(self, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="ambient")
        state = context.state.upper()
        active = state not in ("IDLE",)

        if state in ("FACE_DETECTED", "TRACKING"):
            volume = self.ambience_volume * (0.35 + context.eye_lock_strength * 0.25)
        elif state == "STARE_DETECTED":
            volume = self.ambience_volume * (0.42 + context.eye_lock_strength * 0.32)
        elif state == "ACTIVATING":
            volume = self.ambience_volume * (0.48 + context.progress / 100.0 * 0.22)
        elif state == "POWER_ACTIVE":
            volume = self.ambience_volume * 0.72
        elif state == "INTERRUPTED":
            volume = self.ambience_volume * 0.42
        elif state == "COOLDOWN":
            volume = self.ambience_volume * 0.22
        else:
            volume = 0.0
            active = False

        plan.layer_targets[AudioLayer.AMBIENCE] = LayerTarget(
            layer=AudioLayer.AMBIENCE,
            cue=self._ambient_drone(context),
            volume=clamp(volume),
            pan=self._pan(context),
            fade_ms=self.transition_fade_ms,
            active=active,
        )
        return plan

    def _pan(self, context: AudioEventContext) -> float:
        return clamp(
            (context.eye_lock_strength - 0.5) * self.spatialization_intensity * 0.5,
            -1.0,
            1.0,
        )

    def _pressure_breath(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "ambience_pressure",
            context,
            880.0,
            88.0,
            92.0,
            0.015,
            0.22,
            "pressure breath",
        )

    def _ambient_drone(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "ambience_drone",
            context,
            3000.0,
            32.0,
            34.0,
            0.01,
            0.28,
            "ambient drone",
            loop=True,
        )

    def _make_cue(
        self,
        cue_id: str,
        context: AudioEventContext,
        duration_ms: float,
        start_freq: float,
        end_freq: float,
        noise: float,
        volume_scale: float,
        description: str,
        loop: bool = False,
    ) -> AudioCue:
        samples = self._synthesize_drone(
            duration_ms,
            start_freq,
            end_freq,
            volume_scale,
            noise,
            self._pan(context),
            loop,
        )
        return AudioCue(
            cue_id=cue_id,
            layer=AudioLayer.AMBIENCE,
            event_name=context.event_name,
            duration_ms=duration_ms,
            volume=self.ambience_volume * volume_scale,
            pan=self._pan(context),
            loop=loop,
            fade_in_ms=self.transition_fade_ms if loop else 0.0,
            fade_out_ms=self.transition_fade_ms,
            samples=samples,
            description=description,
        )

    def _synthesize_drone(
        self,
        duration_ms: float,
        start_freq: float,
        end_freq: float,
        volume: float,
        noise: float,
        pan: float,
        loop: bool,
    ) -> np.ndarray:
        sample_rate = 44100
        seconds = max(0.12, duration_ms / 1000.0)
        sample_count = max(1, int(sample_rate * seconds))
        timeline = np.linspace(0.0, seconds, sample_count, endpoint=False)
        sweep = np.linspace(start_freq, end_freq, sample_count)
        phase = 2.0 * math.pi * np.cumsum(sweep) / sample_rate
        rng = np.random.default_rng(int(start_freq * 11.0 + duration_ms))
        carrier = (
            np.sin(phase)
            + 0.22 * np.sin(phase * 0.5)
            + rng.normal(0.0, noise, sample_count)
        )
        carrier += 0.05 * np.sin(timeline * math.pi * 0.45)

        attack = max(1, int(sample_rate * 0.03))
        release = max(1, int(sample_rate * (0.08 if loop else 0.22)))
        envelope = np.ones(sample_count)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        if loop:
            envelope = 0.6 + 0.4 * np.sin(np.minimum(1.0, timeline / seconds) * math.pi)

        stereo_pan = np.clip(pan, -1.0, 1.0)
        left = carrier * envelope * volume * (1.0 - max(0.0, stereo_pan))
        right = carrier * envelope * volume * (1.0 + min(0.0, stereo_pan))
        return np.clip(np.stack([left, right], axis=-1), -1.0, 1.0)
