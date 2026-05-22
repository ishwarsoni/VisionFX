"""Top-level facade orchestrating all Phase 7 recording subsystems."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import RecordingConfig
from recording.export_manager import ExportManager
from recording.performance_monitor import PerformanceMonitor
from recording.quality_manager import QualityManager, QualityMode
from recording.recording_data import (
    ExportInfo,
    RecordingState,
    RecordingStats,
    ReplayClip,
)
from recording.replay_buffer import ReplayBuffer
from recording.screenshot_manager import ScreenshotManager
from recording.video_recorder import VideoRecorder
from utils.logger import Logger


class RecordingManager:
    """
    Central coordinator for recording, replay, screenshots, quality
    management, and export — follows the same facade pattern as
    ``AudioManager`` and ``EffectManager``.

    The manager receives **final composited frames** from the main loop
    and has **zero** imports from tracking, activation, or effects.
    Activation events are consumed via ``bind_activation_engine`` to
    auto-trigger replay captures.
    """

    def __init__(
        self,
        config: RecordingConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or Logger(name="RecordingManager")

        # Export directory manager
        self.export_manager = ExportManager(
            export_root=config.export_root,
            logger=self.logger,
        )

        # Core subsystems
        self.recorder = VideoRecorder(
            max_queue_depth=config.max_queue_depth,
            logger=self.logger,
        )

        self.replay_buffer = ReplayBuffer(
            duration_s=config.replay_duration_s,
            fps=config.recording_fps,
            enabled=config.replay_buffer_enabled,
            logger=self.logger,
        )

        self.screenshot_manager = ScreenshotManager(
            output_dir=self.export_manager.screenshots_dir,
            quality=config.screenshot_quality,
            image_format=config.screenshot_format,
            logger=self.logger,
        )

        self.quality_manager = QualityManager(
            initial_mode=config.quality_mode,
            logger=self.logger,
        )

        self.perf_monitor = PerformanceMonitor(logger=self.logger)

        # State
        self._creator_mode = config.creator_mode
        self._last_frame: Optional[np.ndarray] = None
        self._resolution: Tuple[int, int] = (1280, 720)

        self.logger.info("RecordingManager initialized successfully")

    # ------------------------------------------------------------------
    # Activation engine binding (event-only, no internal imports)
    # ------------------------------------------------------------------

    def bind_activation_engine(self, activation_engine) -> None:
        """Subscribe to activation events for auto-replay capture."""
        if activation_engine is None:
            return

        activation_engine.on(
            "on_activation_complete",
            lambda payload: self._on_activation_complete(payload),
        )
        self.logger.info("RecordingManager bound to activation engine")

    def _on_activation_complete(self, payload: dict) -> None:
        """Auto-capture a replay clip when an activation completes."""
        if self.replay_buffer.enabled and self.replay_buffer.frame_count > 0:
            clip = self.replay_buffer.capture_replay(
                duration_s=self.config.replay_duration_s,
                source_event="activation_complete",
            )
            if clip.is_valid:
                self._export_replay_clip(clip)

    # ------------------------------------------------------------------
    # Frame ingestion (called every frame from main loop)
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Feed a composited frame to the recorder and replay buffer.
        Must be called every frame from the main loop.
        """
        self._last_frame = frame
        h, w = frame.shape[:2]
        self._resolution = (w, h)

        # Feed replay buffer
        self.replay_buffer.push_frame(frame)

        # Feed active recorder
        if self.recorder.is_recording:
            self.recorder.write_frame(frame)
            self.perf_monitor.tick(
                queue_depth=self.recorder.stats.queue_depth,
                dropped=self.recorder.stats.frames_dropped,
            )

    # ------------------------------------------------------------------
    # Recording controls
    # ------------------------------------------------------------------

    def start_recording(self) -> bool:
        """Start recording the composited output."""
        if self.recorder.is_recording:
            self.logger.warning("Recording already active")
            return False

        path = self.export_manager.get_session_path(self.config.recording_format)
        scale = self.quality_manager.profile.resolution_scale
        w = int(self._resolution[0] * scale)
        h = int(self._resolution[1] * scale)

        self.perf_monitor.reset()
        success = self.recorder.start(
            path=path,
            resolution=(w, h),
            fps=self.config.recording_fps,
            codec=self.config.recording_codec,
        )
        if success:
            self.perf_monitor.recording_active = True
        return success

    def stop_recording(self) -> RecordingStats:
        """Stop recording and finalize the file."""
        self.perf_monitor.recording_active = False
        return self.recorder.stop()

    def toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self.recorder.is_recording:
            stats = self.stop_recording()
            self.logger.info(
                f"Recording saved: {stats.frames_written} frames, "
                f"{stats.duration_s:.1f}s"
            )
        else:
            self.start_recording()

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def take_screenshot(self, frame: Optional[np.ndarray] = None) -> ExportInfo:
        """Capture a screenshot of the current composited frame."""
        target = frame if frame is not None else self._last_frame
        if target is None:
            self.logger.warning("No frame available for screenshot")
            return ExportInfo(success=False, error="No frame")
        return self.screenshot_manager.capture(target)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def capture_replay(self, duration_s: Optional[float] = None) -> ReplayClip:
        """Manually capture a replay clip from the buffer."""
        return self.replay_buffer.capture_replay(
            duration_s=duration_s or self.config.replay_duration_s,
            source_event="manual",
        )

    def export_replay(self, duration_s: Optional[float] = None) -> ExportInfo:
        """Capture and export a replay clip to disk."""
        clip = self.capture_replay(duration_s)
        if not clip.is_valid:
            return ExportInfo(success=False, error="Empty replay buffer")
        return self._export_replay_clip(clip)

    def _export_replay_clip(self, clip: ReplayClip) -> ExportInfo:
        """Write a ReplayClip to disk as a video file."""
        path = self.export_manager.get_replay_path(self.config.recording_format)

        if not clip.frames:
            return ExportInfo(path=path, success=False, error="No frames in clip")

        h, w = clip.frames[0].shape[:2]
        temp_recorder = VideoRecorder(
            max_queue_depth=len(clip.frames) + 10,
            logger=self.logger,
        )

        if not temp_recorder.start(
            path=path,
            resolution=(w, h),
            fps=int(clip.fps),
            codec=self.config.recording_codec,
        ):
            return ExportInfo(path=path, success=False, error="Failed to open writer")

        for frame in clip.frames:
            temp_recorder.write_frame(frame)

        stats = temp_recorder.stop()

        import os

        file_size = os.path.getsize(path) if os.path.exists(path) else 0

        info = ExportInfo(
            path=path,
            format=self.config.recording_format,
            width=w,
            height=h,
            duration_s=clip.duration_s,
            file_size_bytes=file_size,
            frame_count=clip.frame_count,
            success=True,
        )
        self.logger.info(
            f"Replay exported: {clip.frame_count} frames, "
            f"{clip.duration_s:.1f}s → {path}"
        )
        return info

    # ------------------------------------------------------------------
    # Quality mode
    # ------------------------------------------------------------------

    def set_quality_mode(self, mode: str) -> None:
        """Switch quality mode (LOW/MEDIUM/HIGH/CINEMATIC)."""
        self.quality_manager.set_mode(mode)

    # ------------------------------------------------------------------
    # Creator mode
    # ------------------------------------------------------------------

    @property
    def is_creator_mode(self) -> bool:
        return self._creator_mode

    def set_creator_mode(self, enabled: bool) -> None:
        """Toggle creator mode — hides debug overlays for clean footage."""
        self._creator_mode = enabled
        state = "enabled" if enabled else "disabled"
        self.logger.info(f"Creator mode {state}")

    def toggle_creator_mode(self) -> None:
        self.set_creator_mode(not self._creator_mode)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def get_debug_lines(self) -> List[str]:
        """Return recording debug lines for the overlay."""
        if not self.config.enable_debug:
            return []

        lines = ["── RECORDING ──"]

        if self.recorder.is_recording:
            self.recorder.update_stats()
            lines.extend(self.recorder.stats.debug_lines())
        else:
            lines.append("REC: IDLE")

        lines.extend(self.quality_manager.get_debug_lines())

        buf = self.replay_buffer
        lines.append(
            f"REPLAY BUF: {buf.frame_count}/{buf.capacity} "
            f"({buf.fill_ratio * 100:.0f}%)"
        )

        if self._creator_mode:
            lines.append("CREATOR MODE: ON")

        return lines

    def get_recording_stats(self) -> RecordingStats:
        """Get current recording statistics."""
        if self.recorder.is_recording:
            return self.recorder.update_stats()
        return self.recorder.stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release all recording resources."""
        if self.recorder.is_recording:
            self.stop_recording()
        self.recorder.release()
        self.replay_buffer.release()
        self.logger.info("RecordingManager released")
