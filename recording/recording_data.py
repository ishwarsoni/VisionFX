"""Data containers for Phase 7 recording, replay, quality, and export subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Recording state machine
# ---------------------------------------------------------------------------


class RecordingState(Enum):
    """Recording pipeline states."""

    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    FINALIZING = "FINALIZING"


# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------


class QualityMode(Enum):
    """Rendering quality presets."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CINEMATIC = "CINEMATIC"


@dataclass
class QualityProfile:
    """Resolved quality parameters for a given mode."""

    mode: QualityMode = QualityMode.HIGH
    resolution_scale: float = 1.0
    compositor_quality: int = 2
    target_fps: int = 30
    effect_intensity_scale: float = 1.0
    label: str = "HIGH"


# ---------------------------------------------------------------------------
# Recording statistics
# ---------------------------------------------------------------------------


@dataclass
class RecordingStats:
    """Live statistics for an active recording session."""

    state: RecordingState = RecordingState.IDLE
    frames_written: int = 0
    frames_dropped: int = 0
    duration_s: float = 0.0
    file_size_bytes: int = 0
    current_fps: float = 0.0
    queue_depth: int = 0
    max_queue_depth: int = 0
    file_path: str = ""

    def debug_lines(self) -> List[str]:
        lines = [
            f"REC: {self.state.value}",
            f"FRAMES: {self.frames_written} | DROP: {self.frames_dropped}",
            f"TIME: {self.duration_s:.1f}s | FPS: {self.current_fps:.1f}",
            f"QUEUE: {self.queue_depth}/{self.max_queue_depth}",
        ]
        if self.file_path:
            # Show only the filename, not full path
            short = self.file_path.replace("\\", "/").rsplit("/", 1)[-1]
            lines.append(f"FILE: {short}")
        return lines


# ---------------------------------------------------------------------------
# Replay clip
# ---------------------------------------------------------------------------


@dataclass
class ReplayClip:
    """A captured replay segment extracted from the circular buffer."""

    frames: List[np.ndarray] = field(default_factory=list)
    timestamps_ms: List[float] = field(default_factory=list)
    duration_s: float = 0.0
    source_event: str = ""
    fps: float = 30.0

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def is_valid(self) -> bool:
        return len(self.frames) > 0


# ---------------------------------------------------------------------------
# Export information
# ---------------------------------------------------------------------------


@dataclass
class ExportInfo:
    """Metadata about a completed export (recording, replay, or screenshot)."""

    path: str = ""
    format: str = ""
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    file_size_bytes: int = 0
    frame_count: int = 0
    success: bool = False
    error: str = ""
