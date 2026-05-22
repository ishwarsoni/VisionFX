"""Orchestrates the cinematic presentation and immersion layer."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from activation.activation_data import ActivationSnapshot
from activation.activation_engine import ActivationEngine
from config.settings import PresentationConfig
from presentation.activation_sequence import ActivationSequence, SequenceState
from presentation.cinematic_camera import CinematicCamera
from presentation.cinematic_text import CinematicTextSystem
from presentation.corruption_system import CorruptionSystem
from presentation.hud_manager import HUDManager
from presentation.immersion_layer import ImmersionLayer
from presentation.presentation_data import (
    PresentationContext,
    PresentationSnapshot,
    PresentationStage,
)
from tracking.tracking_data import TrackingFrameData
from utils.logger import Logger


class PresentationManager:
    """Event-driven cinematic presentation layer for Phase 5."""

    def __init__(self, config: PresentationConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger(name="PresentationManager")

        self.sequence = ActivationSequence(
            stage_hold_ms=config.stage_hold_ms,
            transition_fade_ms=config.transition_fade_ms,
            transition_pulse_ms=config.transition_pulse_ms,
            transition_glitch_ms=config.transition_glitch_ms,
            transition_zoom_ms=config.transition_zoom_ms,
        )
        self.camera = CinematicCamera(zoom_strength=config.cinematic_zoom_strength)
        self.hud = HUDManager(hud_intensity=config.hud_intensity)
        self.corruption = CorruptionSystem(
            corruption_intensity=config.corruption_intensity
        )
        self.immersion = ImmersionLayer(immersion_strength=config.immersion_strength)
        self.text = CinematicTextSystem(animation_ms=config.text_animation_ms)
        self.snapshot = PresentationSnapshot()
        self.last_update_time_ms: Optional[float] = None
        self.frame_index = 0
        self.render_time_ms = 0.0
        self.debug_enabled = config.enable_debug
        self._sequence_state = SequenceState()

        self.logger.info("PresentationManager initialized successfully")

    def bind_activation_engine(self, activation_engine: ActivationEngine) -> None:
        for event_name in (
            "on_stare_start",
            "on_activation_start",
            "on_activation_complete",
            "on_interrupt",
            "on_cooldown_start",
            "on_cooldown_end",
        ):
            activation_engine.on(
                event_name,
                lambda payload, event_name=event_name: self.handle_event(
                    event_name, payload
                ),
            )

    def handle_event(self, event_name: str, payload: dict) -> None:
        mapped_event = self._map_event(event_name)
        self.sequence.handle_event(mapped_event, payload)
        self.hud.handle_event(mapped_event, payload)
        self.corruption.handle_event(mapped_event, payload)
        self.text.handle_event(mapped_event, payload)

    def update(
        self,
        tracking_data: Optional[TrackingFrameData],
        activation_data: Optional[ActivationSnapshot],
        frame_size: tuple,
    ) -> PresentationContext:
        now_ms = time.perf_counter() * 1000.0
        delta_ms = (
            0.0
            if self.last_update_time_ms is None
            else max(0.0, now_ms - self.last_update_time_ms)
        )
        self.last_update_time_ms = now_ms
        self.frame_index += 1

        if activation_data is None:
            activation_data = ActivationSnapshot()

        self._sequence_state = self.sequence.update(activation_data, delta_ms)
        context = PresentationContext(
            frame_index=self.frame_index,
            timestamp_ms=now_ms,
            delta_ms=delta_ms,
            frame_size=frame_size,
            activation_data=activation_data,
            tracking_data=tracking_data,
            stage=self._sequence_state.stage,
            stage_progress=self._sequence_state.progress,
            activation_progress=activation_data.activation_progress,
            activation_ready=activation_data.activation_ready,
        )

        self.hud.update(context, self._sequence_state)
        self.corruption.update(context, self._sequence_state)
        self.text.update(delta_ms)
        self._refresh_snapshot(context)
        return context

    def render(self, frame: np.ndarray, context: PresentationContext) -> np.ndarray:
        if not self.config.enabled:
            self.snapshot = PresentationSnapshot()
            return frame

        start_ms = time.perf_counter() * 1000.0
        result = frame

        result = self.camera.apply(result, context, self._sequence_state)
        result = self.immersion.apply(result, context, self._sequence_state)
        result = self.hud.render(result, context, self._sequence_state)
        result = self.text.render(result, context, self._sequence_state)
        result = self.corruption.apply(result, context, self._sequence_state)

        self.render_time_ms = time.perf_counter() * 1000.0 - start_ms
        self.snapshot.debug_label = f"{self.render_time_ms:.2f}ms"
        self.snapshot.transition_intensity = (
            self.sequence.transition_engine.current_value
        )
        return result

    def get_debug_lines(self) -> List[str]:
        if not self.config.enable_debug:
            return []
        lines = [f"PRES RENDER: {self.render_time_ms:.2f}ms"]
        lines.extend(self.snapshot.lines())
        if self.snapshot.queued_transitions:
            lines.append(f"QUEUE: {', '.join(self.snapshot.queued_transitions)}")
        return lines

    def release(self) -> None:
        self.sequence.transition_engine.queue.clear()
        self.sequence.transition_engine.current = None

    def _map_event(self, event_name: str) -> str:
        mapping = {
            "on_stare_start": "on_eye_lock",
            "on_activation_start": "on_activation_start",
            "on_activation_complete": "on_power_active",
            "on_interrupt": "on_interrupt",
            "on_cooldown_start": "on_cooldown",
            "on_cooldown_end": "on_cooldown",
        }
        return mapping.get(event_name, event_name)

    def _refresh_snapshot(self, context: PresentationContext) -> None:
        activation_data = context.activation_data or ActivationSnapshot()
        self.snapshot.stage = self._sequence_state.stage
        self.snapshot.stage_progress = self._sequence_state.progress
        self.snapshot.hud_intensity = self.hud.current_alpha
        self.snapshot.camera_motion = (
            self.camera.zoom_strength * self._sequence_state.progress
        )
        self.snapshot.corruption_intensity = self.corruption.current_intensity
        self.snapshot.immersion_intensity = (
            self._sequence_state.eye_lock_intensity * self.config.immersion_strength
        )
        self.snapshot.text_intensity = min(1.0, len(self.text.cues) / 3.0)
        self.snapshot.activation_progress = activation_data.activation_progress
        self.snapshot.activation_state = activation_data.state
        self.snapshot.interruption_reason = activation_data.interruption_reason
        self.snapshot.active_layers = self._active_layers()
        self.snapshot.queued_transitions = [
            item.name for item in self.sequence.transition_engine.queue
        ]

    def _active_layers(self) -> List[str]:
        layers: List[str] = []
        if self._sequence_state.eye_lock_intensity > 0.05:
            layers.extend(["hud", "camera"])
        if self._sequence_state.buildup_intensity > 0.05:
            layers.append("immersion")
        if self._sequence_state.corruption_intensity > 0.05:
            layers.append("corruption")
        if self._sequence_state.transform_intensity > 0.05:
            layers.append("text")
        return layers
