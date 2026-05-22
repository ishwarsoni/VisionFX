"""Queued audio synchronization utilities."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import List

from audio.audio_data import AudioCue


@dataclass(order=True)
class ScheduledAudioCue:
    playback_at_ms: float
    priority: int
    cue: AudioCue = field(compare=False)


class AudioSyncEngine:
    def __init__(self) -> None:
        self._queue: List[ScheduledAudioCue] = []
        self.last_dispatch_ms: float = 0.0
        self.last_drift_ms: float = 0.0

    def schedule(self, cue: AudioCue, lead_ms: float = 0.0, priority: int = 0) -> None:
        scheduled_ms = max(cue.timestamp_ms, self.now_ms()) + max(0.0, lead_ms)
        cue.scheduled_ms = scheduled_ms
        heapq.heappush(self._queue, ScheduledAudioCue(scheduled_ms, -priority, cue))

    def drain_ready(self, now_ms: float | None = None) -> List[AudioCue]:
        current_ms = self.now_ms() if now_ms is None else now_ms
        ready: List[AudioCue] = []

        while self._queue and self._queue[0].playback_at_ms <= current_ms:
            scheduled = heapq.heappop(self._queue)
            ready.append(scheduled.cue)
            self.last_dispatch_ms = current_ms
            self.last_drift_ms = current_ms - scheduled.playback_at_ms

        return ready

    def clear(self) -> None:
        self._queue.clear()
        self.last_dispatch_ms = 0.0
        self.last_drift_ms = 0.0

    def pending_count(self) -> int:
        return len(self._queue)

    def pending_labels(self) -> List[str]:
        return [item.cue.label() for item in self._queue[:5]]

    @staticmethod
    def now_ms() -> float:
        return time.perf_counter() * 1000.0
