"""Backend-agnostic sound playback with an optional pygame mixer backend."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from audio.audio_data import AudioCue
from utils.logger import Logger


@dataclass
class SoundHandle:
    cue_id: str
    category: str
    channel: Any
    volume: float = 1.0
    pan: float = 0.0


class SoundPlayer:
    def __init__(
        self,
        asset_root: str,
        master_volume: float = 1.0,
        backend: str = "auto",
        enabled: bool = True,
        logger: Optional[Logger] = None,
    ) -> None:
        self.asset_root = asset_root
        self.master_volume = max(0.0, min(1.0, master_volume))
        self.backend = backend
        self.enabled = enabled
        self.logger = logger or Logger(name="SoundPlayer")

        self._pygame = None
        self._ready = False
        self._initialize_backend()

    @property
    def is_ready(self) -> bool:
        return self._ready and self.enabled

    def _initialize_backend(self) -> None:
        requested_backend = self.backend

        if not self.enabled:
            self.backend = "disabled"
            return

        if requested_backend not in ("auto", "pygame"):
            self.backend = "silent"
            return

        try:
            pygame = importlib.import_module("pygame")
            pygame.mixer.pre_init(44100, -16, 2, 256)
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._pygame = pygame
            self.backend = "pygame"
            self._ready = True
            self.logger.info("Audio backend initialized: pygame")
        except Exception as exc:
            self._pygame = None
            self._ready = False
            self.backend = "silent"
            if requested_backend == "pygame":
                self.logger.debug(f"Audio backend unavailable: {exc}")

    def play_cue(self, cue: AudioCue) -> Optional[SoundHandle]:
        if not self.is_ready or self._pygame is None:
            return None

        sound = None
        if cue.samples is not None:
            sound = self._build_sound_from_samples(cue.samples)
        elif cue.sound_path:
            path = cue.sound_path
            if not os.path.isabs(path):
                path = os.path.join(self.asset_root, path)
            if os.path.exists(path):
                sound = self._pygame.mixer.Sound(path)

        if sound is None:
            return None

        channel = sound.play(
            loops=-1 if cue.loop else 0, fade_ms=max(0, int(cue.fade_in_ms))
        )
        if channel is None:
            return None

        left_volume, right_volume = self._pan_to_stereo(
            cue.volume * self.master_volume, cue.pan
        )
        channel.set_volume(left_volume, right_volume)
        return SoundHandle(
            cue_id=cue.cue_id,
            category=cue.layer.value,
            channel=channel,
            volume=cue.volume,
            pan=cue.pan,
        )

    def set_handle_volume(
        self, handle: Optional[SoundHandle], volume: float, pan: Optional[float] = None
    ) -> None:
        if handle is None or handle.channel is None:
            return

        current_pan = handle.pan if pan is None else pan
        left_volume, right_volume = self._pan_to_stereo(
            volume * self.master_volume, current_pan
        )
        handle.channel.set_volume(left_volume, right_volume)
        handle.volume = volume
        handle.pan = current_pan

    def fadeout_handle(self, handle: Optional[SoundHandle], fade_ms: float) -> None:
        if handle is None or handle.channel is None:
            return

        try:
            handle.channel.fadeout(max(0, int(fade_ms)))
        except Exception:
            try:
                handle.channel.stop()
            except Exception:
                pass

    def stop_handle(self, handle: Optional[SoundHandle]) -> None:
        if handle is None or handle.channel is None:
            return

        try:
            handle.channel.stop()
        except Exception:
            pass

    def stop_all(self) -> None:
        if self._pygame is None:
            return

        try:
            self._pygame.mixer.stop()
        except Exception:
            pass

    def release(self) -> None:
        self.stop_all()
        if self._pygame is not None:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass
        self._ready = False

    def _build_sound_from_samples(self, samples: Any):
        if self._pygame is None:
            return None

        array = np.asarray(samples)
        if array.ndim == 1:
            array = np.stack([array, array], axis=-1)
        if array.dtype != np.int16:
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767.0).astype(np.int16)
        return self._pygame.sndarray.make_sound(array.copy(order="C"))

    @staticmethod
    def _pan_to_stereo(volume: float, pan: float) -> tuple[float, float]:
        pan = max(-1.0, min(1.0, pan))
        left = volume * (1.0 - max(0.0, pan))
        right = volume * (1.0 + min(0.0, pan))
        return max(0.0, min(1.0, left)), max(0.0, min(1.0, right))
