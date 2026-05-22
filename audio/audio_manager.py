"""Integration facade for Phase 6 audio and synchronization."""

from __future__ import annotations

from typing import Dict, List, Optional

from audio.audio_data import AudioSnapshot
from audio.audio_engine import AudioEngine
from config.settings import AudioConfig
from utils.logger import Logger


class AudioManager:
    def __init__(self, config: AudioConfig, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = logger or Logger(name="AudioManager")
        self.engine = AudioEngine(config=config, logger=self.logger)

    def bind_activation_engine(self, activation_engine) -> None:
        if activation_engine is None:
            return

        for event_name in (
            "on_stare_start",
            "on_eye_lock",
            "on_stare_detected",
            "on_activation_start",
            "on_activation_complete",
            "on_interrupt",
            "on_cooldown_start",
            "on_cooldown_end",
        ):
            activation_engine.on(
                event_name,
                lambda payload, name=event_name: self.handle_event(name, payload),
            )

    def handle_event(
        self, event_name: str, payload: Dict[str, object]
    ) -> AudioSnapshot:
        return self.engine.handle_event(event_name, payload)

    def update(self, activation_snapshot) -> AudioSnapshot:
        return self.engine.update(activation_snapshot)

    def get_debug_lines(self) -> List[str]:
        if not self.config.enable_debug:
            return []
        return self.engine.get_debug_lines()

    def is_active(self) -> bool:
        return self.config.enabled and self.engine.is_ready

    def release(self) -> None:
        self.engine.release()
