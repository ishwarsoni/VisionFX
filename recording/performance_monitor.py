"""Lightweight performance monitoring for the recording subsystem."""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional

from utils.logger import Logger


class PerformanceMonitor:
    """
    Tracks recording-specific performance metrics: write FPS, queue
    pressure, dropped frames, and estimated memory usage.
    """

    def __init__(self, window: int = 60, logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger(name="PerformanceMonitor")
        self._window = max(10, window)
        self._fps_samples: deque = deque(maxlen=self._window)
        self._last_tick: float = time.perf_counter()

        # Cumulative counters
        self.total_frames: int = 0
        self.total_dropped: int = 0
        self.peak_queue_depth: int = 0
        self.current_queue_depth: int = 0
        self.recording_active: bool = False

    def tick(self, queue_depth: int = 0, dropped: int = 0) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        if dt > 0:
            self._fps_samples.append(1.0 / dt)

        self.total_frames += 1
        self.total_dropped = dropped
        self.current_queue_depth = queue_depth
        self.peak_queue_depth = max(self.peak_queue_depth, queue_depth)

    @property
    def avg_fps(self) -> float:
        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)

    def get_debug_lines(self) -> List[str]:
        lines = [
            f"REC FPS: {self.avg_fps:.1f}",
            f"QUEUE: {self.current_queue_depth} (peak {self.peak_queue_depth})",
            f"DROPPED: {self.total_dropped}",
            f"TOTAL: {self.total_frames}",
        ]
        return lines

    def reset(self) -> None:
        self._fps_samples.clear()
        self.total_frames = 0
        self.total_dropped = 0
        self.peak_queue_depth = 0
        self.current_queue_depth = 0
        self._last_tick = time.perf_counter()
