"""Threaded video recorder that writes composited frames to disk without blocking the main loop."""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from recording.recording_data import RecordingState, RecordingStats
from utils.logger import Logger

# Sentinel value to signal the writer thread to stop
_STOP_SENTINEL = None


class VideoRecorder:
    """
    Asynchronous video writer backed by a background thread.

    Frames are pushed into a bounded queue from the main loop.  A dedicated
    writer thread drains the queue and feeds ``cv2.VideoWriter``.  If the
    queue is full when ``write_frame`` is called the frame is dropped and the
    drop counter incremented — the main loop is never blocked.
    """

    # Codec lookup
    CODEC_MAP = {
        "mp4v": "mp4v",
        "avc1": "avc1",
        "xvid": "XVID",
        "mjpg": "MJPG",
    }

    def __init__(
        self,
        max_queue_depth: int = 90,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="VideoRecorder")
        self.max_queue_depth = max(10, max_queue_depth)

        # Writer state
        self._writer: Optional[cv2.VideoWriter] = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.max_queue_depth)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Recording bookkeeping
        self.state = RecordingState.IDLE
        self.stats = RecordingStats()
        self._start_time: float = 0.0
        self._file_path: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        path: str,
        resolution: Tuple[int, int],
        fps: int = 30,
        codec: str = "mp4v",
    ) -> bool:
        """
        Open the video writer and start the background write thread.

        Args:
            path: Output file path.
            resolution: (width, height) of frames that will be written.
            fps: Frames per second for the output file.
            codec: FourCC codec string (e.g. ``mp4v``, ``xvid``).

        Returns:
            True if the writer was opened successfully.
        """
        with self._lock:
            if self.state == RecordingState.RECORDING:
                self.logger.warning("Recording already active")
                return False

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            fourcc_str = self.CODEC_MAP.get(codec.lower(), codec)
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            width, height = resolution

            self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if not self._writer.isOpened():
                self.logger.error(f"Failed to open VideoWriter: {path}")
                self._writer = None
                return False

            self._file_path = path
            self._start_time = time.time()

            # Reset stats
            self.stats = RecordingStats(
                state=RecordingState.RECORDING,
                max_queue_depth=self.max_queue_depth,
                file_path=path,
            )

            # Clear the queue in case of leftover frames
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            # Start writer thread
            self.state = RecordingState.RECORDING
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="VideoRecorderWriter",
                daemon=True,
            )
            self._thread.start()

            self.logger.info(f"Recording started: {path} ({width}x{height} @ {fps}fps)")
            return True

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Enqueue a frame for writing.  Non-blocking — drops the frame if the
        queue is full.

        Returns:
            True if the frame was enqueued, False if dropped.
        """
        if self.state != RecordingState.RECORDING:
            return False

        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            self.stats.frames_dropped += 1
            return False

    def stop(self) -> RecordingStats:
        """
        Stop recording, drain remaining frames, and release the writer.

        Returns:
            Final ``RecordingStats`` snapshot.
        """
        with self._lock:
            if self.state != RecordingState.RECORDING:
                return self.stats

            self.state = RecordingState.FINALIZING
            self.stats.state = RecordingState.FINALIZING

        # Signal the writer thread to finish
        self._queue.put(_STOP_SENTINEL)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        # Release writer
        if self._writer:
            self._writer.release()
            self._writer = None

        # Final stats
        self.stats.duration_s = time.time() - self._start_time
        if os.path.exists(self._file_path):
            self.stats.file_size_bytes = os.path.getsize(self._file_path)

        self.state = RecordingState.IDLE
        self.stats.state = RecordingState.IDLE

        self.logger.info(
            f"Recording stopped: {self.stats.frames_written} frames, "
            f"{self.stats.duration_s:.1f}s, "
            f"{self.stats.frames_dropped} dropped"
        )
        return self.stats

    def update_stats(self) -> RecordingStats:
        """Refresh live stats (called each frame from the main loop)."""
        if self.state == RecordingState.RECORDING:
            self.stats.duration_s = time.time() - self._start_time
            self.stats.queue_depth = self._queue.qsize()
            if self.stats.duration_s > 0:
                self.stats.current_fps = (
                    self.stats.frames_written / self.stats.duration_s
                )
        return self.stats

    @property
    def is_recording(self) -> bool:
        return self.state == RecordingState.RECORDING

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Drain the queue and write frames to disk."""
        while True:
            try:
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self.state != RecordingState.RECORDING:
                    break
                continue

            if frame is _STOP_SENTINEL:
                # Drain remaining
                while not self._queue.empty():
                    try:
                        remaining = self._queue.get_nowait()
                        if remaining is not _STOP_SENTINEL and self._writer:
                            self._writer.write(remaining)
                            self.stats.frames_written += 1
                    except queue.Empty:
                        break
                break

            if self._writer:
                self._writer.write(frame)
                self.stats.frames_written += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Force-release all resources."""
        if self.state == RecordingState.RECORDING:
            self.stop()
        if self._writer:
            self._writer.release()
            self._writer = None
