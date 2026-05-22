"""Cinematic activation sequencing and stage pacing."""

from __future__ import annotations

from dataclasses import dataclass

from activation.activation_data import ActivationSnapshot
from activation.activation_state import ActivationState
from presentation.presentation_data import PresentationStage
from presentation.transition_engine import TransitionEngine


@dataclass
class SequenceState:
    stage: PresentationStage = PresentationStage.IDLE
    progress: float = 0.0
    stage_progress: float = 0.0
    stage_time_ms: float = 0.0
    eye_lock_intensity: float = 0.0
    buildup_intensity: float = 0.0
    corruption_intensity: float = 0.0
    transform_intensity: float = 0.0
    interruption_intensity: float = 0.0
    cooldown_intensity: float = 0.0


class ActivationSequence:
    """Maps activation state progression to cinematic presentation pacing."""

    def __init__(
        self,
        stage_hold_ms: float,
        transition_fade_ms: float,
        transition_pulse_ms: float,
        transition_glitch_ms: float,
        transition_zoom_ms: float,
    ):
        self.stage_hold_ms = stage_hold_ms
        self.transition_engine = TransitionEngine()
        self.state = SequenceState()
        self._last_stage = PresentationStage.IDLE
        self.transition_durations = {
            "fade": transition_fade_ms,
            "pulse": transition_pulse_ms,
            "glitch": transition_glitch_ms,
            "zoom": transition_zoom_ms,
        }

    def handle_event(self, event_name: str, payload: dict) -> None:
        if event_name in ("on_eye_lock", "on_stare_detected"):
            self.transition_engine.queue_transition(
                "eye_lock", "pulse", self.transition_durations["pulse"], 0.0, 1.0
            )
        elif event_name == "on_activation_start":
            self.transition_engine.queue_transition(
                "activation_start", "zoom", self.transition_durations["zoom"], 0.0, 1.0
            )
        elif event_name == "on_power_active":
            self.transition_engine.queue_transition(
                "power_active", "glitch", self.transition_durations["glitch"], 0.0, 1.0
            )
        elif event_name == "on_interrupt":
            self.transition_engine.interrupt(
                "interrupt", "fade", self.transition_durations["fade"]
            )
        elif event_name == "on_cooldown":
            self.transition_engine.queue_transition(
                "cooldown", "fade", self.transition_durations["fade"], 1.0, 0.0
            )

    def update(
        self, activation_data: ActivationSnapshot, delta_ms: float
    ) -> SequenceState:
        self.transition_engine.update(delta_ms)

        stage = self._derive_stage(activation_data)
        if stage != self._last_stage:
            self.state.stage_time_ms = 0.0
            self._last_stage = stage
            self._queue_stage_transition(stage)
        else:
            self.state.stage_time_ms += max(0.0, delta_ms)

        stage_progress = self._compute_stage_progress(stage, activation_data)
        self.state.stage = stage
        self.state.progress = stage_progress
        self.state.stage_progress = stage_progress
        self.state.eye_lock_intensity = self._smooth_value(
            stage_progress
            if stage in (PresentationStage.EYE_LOCK, PresentationStage.BUILDUP)
            else 0.0
        )
        self.state.buildup_intensity = self._smooth_value(
            max(0.0, (activation_data.activation_progress / 100.0) - 0.1)
            if activation_data
            else 0.0
        )
        self.state.corruption_intensity = self._smooth_value(
            max(0.0, (activation_data.activation_progress / 100.0) - 0.35)
            if activation_data
            else 0.0
        )
        self.state.transform_intensity = self._smooth_value(
            1.0 if stage == PresentationStage.TRANSFORM else 0.0
        )
        self.state.interruption_intensity = self._smooth_value(
            1.0 if stage == PresentationStage.INTERRUPTED else 0.0
        )
        self.state.cooldown_intensity = self._smooth_value(
            1.0 if stage == PresentationStage.COOLDOWN else 0.0
        )
        return self.state

    def _queue_stage_transition(self, stage: PresentationStage) -> None:
        if stage == PresentationStage.EYE_LOCK:
            self.transition_engine.queue_transition(
                "eye_lock_stage", "pulse", self.transition_durations["pulse"], 0.0, 1.0
            )
        elif stage == PresentationStage.BUILDUP:
            self.transition_engine.queue_transition(
                "buildup_stage", "zoom", self.transition_durations["zoom"], 0.0, 1.0
            )
        elif stage == PresentationStage.CORRUPTION:
            self.transition_engine.queue_transition(
                "corruption_stage",
                "glitch",
                self.transition_durations["glitch"],
                0.0,
                1.0,
            )
        elif stage == PresentationStage.TRANSFORM:
            self.transition_engine.queue_transition(
                "transform_stage", "zoom", self.transition_durations["zoom"], 0.4, 1.0
            )
        elif stage == PresentationStage.INTERRUPTED:
            self.transition_engine.interrupt(
                "interrupt_stage", "fade", self.transition_durations["fade"]
            )
        elif stage == PresentationStage.COOLDOWN:
            self.transition_engine.queue_transition(
                "cooldown_stage", "fade", self.transition_durations["fade"], 1.0, 0.0
            )

    def _derive_stage(self, activation_data: ActivationSnapshot) -> PresentationStage:
        state = activation_data.state
        if state == ActivationState.IDLE:
            return PresentationStage.IDLE
        if state in (
            ActivationState.FACE_DETECTED,
            ActivationState.TRACKING,
            ActivationState.STARE_DETECTED,
        ):
            return (
                PresentationStage.EYE_LOCK
                if activation_data.activation_progress < 15.0
                else PresentationStage.BUILDUP
            )
        if state == ActivationState.ACTIVATING:
            if activation_data.activation_progress < 35.0:
                return PresentationStage.BUILDUP
            if activation_data.activation_progress < 80.0:
                return PresentationStage.CORRUPTION
            return PresentationStage.TRANSFORM
        if state == ActivationState.POWER_ACTIVE:
            return PresentationStage.TRANSFORM
        if state == ActivationState.INTERRUPTED:
            return PresentationStage.INTERRUPTED
        if state == ActivationState.COOLDOWN:
            return PresentationStage.COOLDOWN
        return PresentationStage.IDLE

    def _compute_stage_progress(
        self, stage: PresentationStage, activation_data: ActivationSnapshot
    ) -> float:
        activation_progress = (
            activation_data.activation_progress if activation_data else 0.0
        )
        if stage == PresentationStage.IDLE:
            return 0.0
        if stage == PresentationStage.EYE_LOCK:
            return min(1.0, activation_progress / 15.0)
        if stage == PresentationStage.BUILDUP:
            return min(1.0, max(0.0, (activation_progress - 10.0) / 45.0))
        if stage == PresentationStage.CORRUPTION:
            return min(1.0, max(0.0, (activation_progress - 35.0) / 45.0))
        if stage == PresentationStage.TRANSFORM:
            return min(1.0, max(0.0, (activation_progress - 70.0) / 30.0))
        if stage == PresentationStage.INTERRUPTED:
            return max(
                0.0, 1.0 - min(1.0, activation_data.interruption_duration_ms / 450.0)
            )
        if stage == PresentationStage.COOLDOWN:
            return max(
                0.0, 1.0 - min(1.0, activation_data.cooldown_remaining_ms / 1800.0)
            )
        return 0.0

    @staticmethod
    def _smooth_value(value: float) -> float:
        return max(0.0, min(1.0, value))
