"""Glitch and corruption audio planning."""

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


class GlitchAudioSystem:
    def __init__(
        self, glitch_volume: float, transition_fade_ms: float, glitch_intensity: float
    ) -> None:
        self.glitch_volume = glitch_volume
        self.transition_fade_ms = transition_fade_ms
        self.glitch_intensity = glitch_intensity
        self._last_bucket = -1

    def on_event(self, event_name: str, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="glitch")
        if event_name in ("on_interrupt", "on_activation_complete"):
            plan.cues.append(
                self._make_cue(
                    "glitch_burst_event",
                    context,
                    420.0,
                    1300.0,
                    180.0,
                    0.26,
                    0.9,
                    "glitch burst",
                )
            )
        elif event_name in ("on_cooldown_start", "on_cooldown_end"):
            plan.cues.append(
                self._make_cue(
                    "glitch_falloff_event",
                    context,
                    520.0,
                    300.0,
                    72.0,
                    0.08,
                    0.26,
                    "signal falloff",
                )
            )
        elif event_name in ("on_stare_start", "on_eye_lock", "on_stare_detected"):
            plan.cues.append(
                self._make_cue(
                    "glitch_ping_event",
                    context,
                    260.0,
                    880.0,
                    196.0,
                    0.18,
                    0.28,
                    "interference ping",
                )
            )
        return plan

    def plan(self, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="glitch")
        state = context.state.upper()
        progress = clamp(context.progress / 100.0)

        if state == "STARE_DETECTED":
            target_volume = self.glitch_volume * (
                0.03 + context.eye_lock_strength * 0.08
            )
            active = True
        elif state == "ACTIVATING":
            target_volume = self.glitch_volume * (
                0.1 + progress * self.glitch_intensity
            )
            active = True
        elif state == "POWER_ACTIVE":
            target_volume = self.glitch_volume * 0.9
            active = True
        elif state == "INTERRUPTED":
            target_volume = self.glitch_volume * 0.72
            active = True
        elif state == "COOLDOWN":
            target_volume = self.glitch_volume * 0.14
            active = True
        else:
            target_volume = 0.0
            active = False

        plan.layer_targets[AudioLayer.GLITCH] = LayerTarget(
            layer=AudioLayer.GLITCH,
            cue=self._glitch_bed(context),
            volume=clamp(target_volume),
            pan=self._pan(context),
            fade_ms=self.transition_fade_ms,
            active=active,
        )

        bucket = self._bucket(context)
        if active and bucket != self._last_bucket:
            if bucket == 1:
                plan.cues.append(
                    self._make_cue(
                        "glitch_ping",
                        context,
                        260.0,
                        880.0,
                        196.0,
                        0.18,
                        0.18,
                        "interference ping",
                    )
                )
            elif bucket == 2:
                plan.cues.append(
                    self._make_cue(
                        "glitch_burst_mid",
                        context,
                        420.0,
                        1300.0,
                        180.0,
                        0.24,
                        0.46,
                        "glitch burst",
                    )
                )
            elif bucket == 3:
                plan.cues.append(
                    self._make_cue(
                        "glitch_burst_high",
                        context,
                        420.0,
                        1600.0,
                        164.0,
                        0.26,
                        0.72,
                        "glitch burst",
                    )
                )
            elif bucket >= 4:
                plan.cues.append(
                    self._make_cue(
                        "glitch_burst_peak",
                        context,
                        420.0,
                        1800.0,
                        160.0,
                        0.28,
                        0.95,
                        "glitch burst",
                    )
                )
            self._last_bucket = bucket

        if state in ("IDLE", "FACE_DETECTED", "TRACKING"):
            self._last_bucket = -1

        return plan

    def _bucket(self, context: AudioEventContext) -> int:
        if context.state.upper() == "INTERRUPTED":
            return 4
        if context.state.upper() == "POWER_ACTIVE":
            return 3
        if context.progress >= 80.0:
            return 3
        if context.progress >= 45.0:
            return 2
        if context.progress >= 18.0:
            return 1
        return 0

    def _pan(self, context: AudioEventContext) -> float:
        return clamp((context.glitch_intensity - 0.5) * 0.4, -1.0, 1.0)

    def _glitch_bed(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "glitch_bed",
            context,
            2200.0,
            110.0,
            170.0,
            0.11,
            0.24,
            "glitch bed",
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
        samples = self._synthesize_noise(
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
            layer=AudioLayer.GLITCH,
            event_name=context.event_name,
            duration_ms=duration_ms,
            volume=self.glitch_volume * volume_scale,
            pan=self._pan(context),
            loop=loop,
            fade_in_ms=self.transition_fade_ms if loop else 0.0,
            fade_out_ms=self.transition_fade_ms,
            samples=samples,
            description=description,
        )

    def _synthesize_noise(
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
        seconds = max(0.06, duration_ms / 1000.0)
        sample_count = max(1, int(sample_rate * seconds))
        timeline = np.linspace(0.0, seconds, sample_count, endpoint=False)
        sweep = np.linspace(start_freq, end_freq, sample_count)
        phase = 2.0 * math.pi * np.cumsum(sweep) / sample_rate
        rng = np.random.default_rng(
            int(start_freq * 17.0 + end_freq * 13.0 + duration_ms)
        )
        carrier = (
            np.sin(phase)
            + 0.55 * np.sin(phase * 1.7)
            + rng.normal(0.0, noise, sample_count)
        )
        carrier += 0.12 * np.sign(np.sin(phase * 8.0))

        attack = max(1, int(sample_rate * 0.01))
        release = max(1, int(sample_rate * (0.05 if loop else 0.18)))
        envelope = np.ones(sample_count)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        if loop:
            envelope = 0.55 + 0.45 * np.sin(
                np.minimum(1.0, timeline / seconds) * math.pi
            )

        stereo_pan = np.clip(pan, -1.0, 1.0)
        left = carrier * envelope * volume * (1.0 - max(0.0, stereo_pan))
        right = carrier * envelope * volume * (1.0 + min(0.0, stereo_pan))
        return np.clip(np.stack([left, right], axis=-1), -1.0, 1.0)
