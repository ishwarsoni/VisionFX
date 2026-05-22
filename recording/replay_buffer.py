"""Circular frame buffer for instant replay of activation moments."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

from recording.recording_data import ReplayClip
from utils.logger import Logger


class ReplayBuffer:
    """
    Maintains a rolling window of recent composited frames so the last
    N seconds can be extracted as a replay clip on demand.

    Uses ``collections.deque`` with a bounded ``maxlen`` for constant-time
    push and automatic eviction of old frames.
    """

    def __init__(
        self,
        duration_s: float = 5.0,
        fps: int = 30,
        enabled: bool = True,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="ReplayBuffer")
        self.enabled = enabled
        self.duration_s = max(1.0, duration_s)
        self.fps = max(1, fps)

        capacity = int(self.duration_s * self.fps)
        self._frames: deque = deque(maxlen=capacity)
        self._timestamps: deque = deque(maxlen=capacity)

        self.logger.info(
            f"ReplayBuffer initialized: {self.duration_s}s @ {self.fps}fps "
            f"(capacity={capacity} frames)"
        )

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def push_frame(
        self, frame: np.ndarray, timestamp_ms: Optional[float] = None
    ) -> None:
        """
        Append a composited frame to the circular buffer.

        Args:
            frame: Final composited BGR frame.
            timestamp_ms: Optional timestamp; defaults to current time.
        """
        if not self.enabled:
            return

        if timestamp_ms is None:
            timestamp_ms = time.perf_counter() * 1000.0

        # Store a copy so the caller can reuse the original array
        self._frames.append(frame.copy())
        self._timestamps.append(timestamp_ms)

    # ------------------------------------------------------------------
    # Replay extraction
    # ------------------------------------------------------------------

    def capture_replay(
        self,
        duration_s: Optional[float] = None,
        source_event: str = "manual",
    ) -> ReplayClip:
        """
        Extract the most recent *duration_s* seconds as a ``ReplayClip``.

        If *duration_s* is ``None`` or exceeds the buffer, the entire
        buffer contents are returned.

        Args:
            duration_s: Seconds of footage to extract (default: full buffer).
            source_event: Label describing what triggered the capture.

        Returns:
            A ``ReplayClip`` with copied frames and timestamps.
        """
        if not self._frames:
            self.logger.warning("Replay capture requested but buffer is empty")
            return ReplayClip(source_event=source_event, fps=self.fps)

        # Determine how many frames to extract
        if duration_s is None or duration_s <= 0:
            count = len(self._frames)
        else:
            count = min(len(self._frames), int(duration_s * self.fps))

        frames = list(self._frames)[-count:]
        timestamps = list(self._timestamps)[-count:]

        actual_duration = 0.0
        if len(timestamps) >= 2:
            actual_duration = (timestamps[-1] - timestamps[0]) / 1000.0

        clip = ReplayClip(
            frames=frames,
            timestamps_ms=timestamps,
            duration_s=actual_duration,
            source_event=source_event,
            fps=self.fps,
        )

        self.logger.info(
            f"Replay captured: {clip.frame_count} frames, "
            f"{clip.duration_s:.1f}s ({source_event})"
        )
        return clip

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._frames)

    @property
    def capacity(self) -> int:
        """Maximum number of frames the buffer can hold."""
        return self._frames.maxlen or 0

    @property
    def fill_ratio(self) -> float:
        """Fraction of the buffer that is filled (0.0–1.0)."""
        cap = self.capacity
        return len(self._frames) / cap if cap > 0 else 0.0

    def clear(self) -> None:
        """Discard all buffered frames."""
        self._frames.clear()
        self._timestamps.clear()

    def release(self) -> None:
        """Release buffer resources."""
        self.clear()
