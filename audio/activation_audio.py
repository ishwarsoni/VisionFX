"""Activation-specific cinematic audio planning."""

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


class ActivationAudioSystem:
    def __init__(
        self,
        activation_volume: float,
        transition_fade_ms: float,
        spatialization_intensity: float,
    ) -> None:
        self.activation_volume = activation_volume
        self.transition_fade_ms = transition_fade_ms
        self.spatialization_intensity = spatialization_intensity
        self._last_bucket = -1

    def on_event(self, event_name: str, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="activation")
        if event_name in ("on_stare_start", "on_eye_lock", "on_stare_detected"):
            plan.cues.append(
                self._make_cue(
                    "activation_eye_lock",
                    context,
                    500.0,
                    76.0,
                    92.0,
                    0.03,
                    0.26,
                    "eye lock intake",
                )
            )
        elif event_name == "on_activation_start":
            plan.cues.append(
                self._make_cue(
                    "activation_start",
                    context,
                    800.0,
                    48.0,
                    58.0,
                    0.06,
                    0.72,
                    "activation buildup",
                )
            )
        elif event_name == "on_activation_complete":
            plan.cues.append(
                self._make_cue(
                    "activation_complete",
                    context,
                    960.0,
                    36.0,
                    120.0,
                    0.09,
                    1.0,
                    "activation impact",
                )
            )
        elif event_name == "on_interrupt":
            plan.cues.append(
                self._make_cue(
                    "activation_interrupt",
                    context,
                    640.0,
                    120.0,
                    40.0,
                    0.14,
                    0.52,
                    "activation interruption",
                )
            )
        elif event_name == "on_cooldown_start":
            plan.cues.append(
                self._make_cue(
                    "activation_cooldown",
                    context,
                    520.0,
                    68.0,
                    34.0,
                    0.02,
                    0.3,
                    "cooldown release",
                )
            )
        return plan

    def plan(self, context: AudioEventContext) -> AudioPlan:
        plan = AudioPlan(mood="activation")
        state = context.state.upper()
        progress = clamp(context.progress / 100.0)
        active = state in ("STARE_DETECTED", "ACTIVATING", "POWER_ACTIVE")

        if state == "STARE_DETECTED":
            target_volume = self.activation_volume * (
                0.12 + context.eye_lock_strength * 0.18
            )
        elif state == "ACTIVATING":
            target_volume = self.activation_volume * (0.24 + progress * 0.72)
        elif state == "POWER_ACTIVE":
            target_volume = self.activation_volume * 0.95
        else:
            target_volume = 0.0

        plan.layer_targets[AudioLayer.ACTIVATION] = LayerTarget(
            layer=AudioLayer.ACTIVATION,
            cue=self._activation_bed(context),
            volume=clamp(target_volume),
            pan=self._pan_for_context(context),
            fade_ms=self.transition_fade_ms,
            active=active,
        )

        bucket = self._bucket(context)
        if active and bucket != self._last_bucket:
            if bucket == 0:
                plan.cues.append(
                    self._make_cue(
                        "activation_rise_soft",
                        context,
                        340.0,
                        60.0,
                        76.0,
                        0.02,
                        0.18,
                        "tension rise",
                    )
                )
            elif bucket == 1:
                plan.cues.append(
                    self._make_cue(
                        "activation_rise",
                        context,
                        360.0,
                        60.0,
                        84.0,
                        0.02,
                        0.32,
                        "tension rise",
                    )
                )
            elif bucket == 2:
                plan.cues.append(self._heartbeat_cue(context))
            else:
                plan.cues.append(self._activation_pulse_cue(context))
            self._last_bucket = bucket

        if state in ("IDLE", "FACE_DETECTED", "TRACKING", "COOLDOWN"):
            self._last_bucket = -1

        return plan

    def _bucket(self, context: AudioEventContext) -> int:
        if context.state.upper() == "POWER_ACTIVE":
            return 4
        if context.progress >= 85.0:
            return 3
        if context.progress >= 55.0:
            return 2
        if context.progress >= 20.0:
            return 1
        return 0

    def _pan_for_context(self, context: AudioEventContext) -> float:
        return clamp(
            (context.eye_lock_strength - 0.5) * self.spatialization_intensity * 2.0,
            -1.0,
            1.0,
        )

    def _activation_bed(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "activation_bed",
            context,
            2400.0,
            40.0,
            46.0,
            0.01,
            0.22,
            "activation bed",
            loop=True,
        )

    def _heartbeat_cue(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "activation_heartbeat",
            context,
            280.0,
            42.0,
            42.0,
            0.03,
            0.48,
            "heartbeat pulse",
        )

    def _activation_pulse_cue(self, context: AudioEventContext) -> AudioCue:
        return self._make_cue(
            "activation_pulse",
            context,
            420.0,
            26.0,
            112.0,
            0.06,
            0.92,
            "activation pulse",
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
        samples = self._synthesize_tone(
            duration_ms,
            start_freq,
            end_freq,
            volume_scale,
            noise,
            self._pan_for_context(context),
            loop,
        )
        return AudioCue(
            cue_id=cue_id,
            layer=AudioLayer.ACTIVATION,
            event_name=context.event_name,
            duration_ms=duration_ms,
            volume=self.activation_volume * volume_scale,
            pan=self._pan_for_context(context),
            loop=loop,
            fade_in_ms=self.transition_fade_ms if loop else 0.0,
            fade_out_ms=self.transition_fade_ms,
            samples=samples,
            description=description,
        )

    def _synthesize_tone(
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
        seconds = max(0.08, duration_ms / 1000.0)
        sample_count = max(1, int(sample_rate * seconds))
        timeline = np.linspace(0.0, seconds, sample_count, endpoint=False)
        sweep = np.linspace(start_freq, end_freq, sample_count)
        phase = 2.0 * math.pi * np.cumsum(sweep) / sample_rate
        rng = np.random.default_rng(
            int(start_freq * 1000.0 + end_freq * 10.0 + duration_ms)
        )
        carrier = (
            np.sin(phase) + 0.32 * np.sin(phase * 2.0) + 0.14 * np.sin(phase * 0.5)
        )
        carrier += rng.normal(0.0, noise, sample_count)

        attack = max(1, int(sample_rate * 0.02))
        release = max(1, int(sample_rate * (0.08 if loop else 0.18)))
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
