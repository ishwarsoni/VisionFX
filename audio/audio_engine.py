"""Central audio synchronization and layering engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from audio.activation_audio import ActivationAudioSystem
from audio.ambience_system import AmbienceSystem
from audio.audio_data import (
    AudioCue,
    AudioEventContext,
    AudioLayer,
    AudioPlan,
    AudioSnapshot,
    LayerTarget,
)
from audio.audio_sync import AudioSyncEngine
from audio.audio_transition import VolumeEnvelope, clamp
from audio.glitch_audio import GlitchAudioSystem
from audio.sound_player import SoundHandle, SoundPlayer
from config.settings import AudioConfig
from utils.logger import Logger


@dataclass
class _LayerRuntime:
    target: Optional[LayerTarget] = None
    handle: Optional[SoundHandle] = None
    envelope: VolumeEnvelope = field(default_factory=VolumeEnvelope)


class AudioEngine:
    def __init__(self, config: AudioConfig, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = logger or Logger(name="AudioEngine")

        self.player = SoundPlayer(
            asset_root=config.asset_root,
            master_volume=config.master_volume,
            backend=config.backend,
            enabled=config.enabled,
            logger=self.logger,
        )
        self.sync = AudioSyncEngine()
        self.activation_audio = ActivationAudioSystem(
            config.activation_volume,
            config.transition_fade_ms,
            config.spatialization_intensity,
        )
        self.glitch_audio = GlitchAudioSystem(
            config.glitch_volume, config.transition_fade_ms, config.glitch_intensity
        )
        self.ambience_system = AmbienceSystem(
            config.ambience_volume,
            config.transition_fade_ms,
            config.spatialization_intensity,
        )

        self.layers: Dict[AudioLayer, _LayerRuntime] = {
            layer: _LayerRuntime() for layer in AudioLayer
        }
        self.snapshot = AudioSnapshot(
            enabled=config.enabled,
            backend=self.player.backend,
            master_volume=config.master_volume,
        )
        self.last_context = AudioEventContext()
        self._last_update_ms: Optional[float] = None

        self.logger.info("AudioEngine initialized successfully")

    @property
    def is_ready(self) -> bool:
        return self.config.enabled and self.player.is_ready

    def handle_event(
        self, event_name: str, payload: Dict[str, object]
    ) -> AudioSnapshot:
        context = AudioEventContext.from_payload(event_name, payload)
        self.last_context = context
        self._apply_plans(
            [
                self.activation_audio.on_event(event_name, context),
                self.glitch_audio.on_event(event_name, context),
                self.ambience_system.on_event(event_name, context),
            ],
            context.timestamp_ms,
        )
        self._dispatch_ready_cues(context.timestamp_ms)
        self._advance_layers(context.timestamp_ms)
        self._refresh_snapshot(context)
        return self.snapshot

    def update(self, activation_snapshot: Optional[object]) -> AudioSnapshot:
        if not self.config.enabled:
            self.snapshot.enabled = False
            self.snapshot.backend = self.player.backend
            return self.snapshot

        if activation_snapshot is None:
            context = self.last_context
        else:
            context = AudioEventContext.from_snapshot(activation_snapshot)

        now_ms = context.timestamp_ms
        delta_ms = (
            0.0
            if self._last_update_ms is None
            else max(0.0, now_ms - self._last_update_ms)
        )
        self._last_update_ms = now_ms

        self.last_context = context
        self._apply_plans(
            [
                self.activation_audio.plan(context),
                self.glitch_audio.plan(context),
                self.ambience_system.plan(context),
            ],
            now_ms,
        )
        self._dispatch_ready_cues(now_ms)
        self._advance_layers(now_ms, delta_ms)
        self._refresh_snapshot(context)
        return self.snapshot

    def release(self) -> None:
        self.player.stop_all()
        self.sync.clear()
        self.player.release()

    def get_debug_lines(self) -> List[str]:
        return self.snapshot.debug_lines()

    def _apply_plans(self, plans: Iterable[AudioPlan], now_ms: float) -> None:
        merged = AudioPlan()
        for plan in plans:
            merged.merge(plan)

        for cue in merged.cues:
            cue.timestamp_ms = now_ms
            self.sync.schedule(
                cue, lead_ms=self.config.sync_lead_ms, priority=cue.priority
            )

        for layer, target in merged.layer_targets.items():
            runtime = self.layers[layer]
            runtime.target = target
            runtime.envelope.set_target(clamp(target.volume), target.fade_ms)
            if target.active and target.cue is not None:
                self._ensure_layer_handle(layer, target, now_ms)
            elif not target.active and runtime.handle is not None:
                self.player.fadeout_handle(
                    runtime.handle, target.fade_ms or self.config.transition_fade_ms
                )
                runtime.handle = None

    def _ensure_layer_handle(
        self, layer: AudioLayer, target: LayerTarget, now_ms: float
    ) -> None:
        runtime = self.layers[layer]
        cue = target.cue
        if cue is None:
            return

        if runtime.handle is None or runtime.handle.cue_id != cue.cue_id:
            if runtime.handle is not None:
                self.player.fadeout_handle(
                    runtime.handle, target.fade_ms or self.config.transition_fade_ms
                )
            cue.timestamp_ms = now_ms
            cue.loop = True
            runtime.handle = self.player.play_cue(cue)

        if runtime.handle is not None:
            self.player.set_handle_volume(
                runtime.handle, runtime.envelope.current, target.pan
            )

    def _dispatch_ready_cues(self, now_ms: float) -> None:
        for cue in self.sync.drain_ready(now_ms):
            self.player.play_cue(cue)

    def _advance_layers(self, now_ms: float, delta_ms: float = 16.0) -> None:
        active_layers: List[str] = []
        for layer, runtime in self.layers.items():
            if runtime.target is None:
                continue
            if runtime.handle is not None:
                runtime.envelope.update(delta_ms)
                self.player.set_handle_volume(
                    runtime.handle, runtime.envelope.current, runtime.target.pan
                )
            if runtime.target.active:
                active_layers.append(layer.value)

        self.snapshot.active_layers = active_layers
        self.snapshot.queued_events = self.sync.pending_count()
        self.snapshot.pending_cues = self.sync.pending_labels()
        self.snapshot.sync_lag_ms = self.sync.last_drift_ms

    def _refresh_snapshot(self, context: AudioEventContext) -> None:
        self.snapshot.enabled = self.config.enabled
        self.snapshot.backend = self.player.backend
        self.snapshot.master_volume = self.config.master_volume
        self.snapshot.current_mood = self._infer_mood(context)
        self.snapshot.current_state = context.state
        self.snapshot.current_progress = context.progress
        self.snapshot.last_event = context.event_name

    def _infer_mood(self, context: AudioEventContext) -> str:
        state = context.state.upper()
        if state == "POWER_ACTIVE":
            return "explosive"
        if state == "ACTIVATING":
            return "tense"
        if state == "STARE_DETECTED":
            return "charged"
        if state == "INTERRUPTED":
            return "fractured"
        if state == "COOLDOWN":
            return "resolving"
        if state in ("FACE_DETECTED", "TRACKING"):
            return "watchful"
        return "idle"
