"""
Data structures for tracking information.
Defines types and containers for all tracking-related data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class GazeDirection(Enum):
    """Enumeration for gaze direction estimation."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    CENTER = "CENTER"
    UP = "UP"
    DOWN = "DOWN"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class TrackingState(Enum):
    """Enumeration for overall tracking state."""

    NO_FACE = "NO_FACE"
    FACE_DETECTED = "FACE_DETECTED"
    TRACKING = "TRACKING"
    EYE_CONTACT = "EYE_CONTACT"
    BLINKING = "BLINKING"


@dataclass
class IrisData:
    """Iris position and gaze data for one eye."""

    center: Tuple[float, float] = (0.0, 0.0)  # (x, y) normalized coordinates
    radius: float = 0.0  # Approximate radius
    gaze_direction: GazeDirection = GazeDirection.CENTER
    confidence: float = 0.0  # 0.0 to 1.0

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class EyeData:
    """Comprehensive data for one eye."""

    iris: IrisData = field(default_factory=IrisData)
    landmarks: List[Tuple[float, float]] = field(
        default_factory=list
    )  # Eye region landmarks
    openness: float = 0.0  # 0.0 (closed) to 1.0 (wide open)
    aspect_ratio: float = 0.0  # Eye aspect ratio for blink detection

    def __post_init__(self):
        self.openness = max(0.0, min(1.0, self.openness))


@dataclass
class BlinkData:
    """Blink detection information."""

    is_blinking: bool = False
    blink_start_frame: Optional[int] = None
    blink_duration_ms: float = 0.0
    eye_openness: float = 1.0  # Current eye openness
    blink_count: int = 0
    last_blink_frame: Optional[int] = None

    def get_blink_ratio(self) -> float:
        """Get blink intensity as ratio (0=fully open, 1=fully closed)."""
        return 1.0 - self.eye_openness


@dataclass
class FaceData:
    """Comprehensive face tracking information."""

    detected: bool = False
    center: Tuple[float, float] = (0.0, 0.0)  # Face center (x, y)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    landmarks: List[Tuple[float, float]] = field(
        default_factory=list
    )  # 468 MediaPipe landmarks
    confidence: float = 0.0  # Detection confidence 0.0-1.0
    rotation_degrees: float = 0.0  # Approximate rotation

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class EyeContactData:
    """Eye contact and attention information."""

    in_contact: bool = False
    confidence: float = 0.0
    duration_ms: float = 0.0  # How long gaze has been centered
    iris_symmetry: float = 0.0  # Left-right iris symmetry (0.0-1.0)
    gaze_centered_ratio: float = 0.0  # How centered the gaze is (0.0-1.0)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.iris_symmetry = max(0.0, min(1.0, self.iris_symmetry))
        self.gaze_centered_ratio = max(0.0, min(1.0, self.gaze_centered_ratio))


@dataclass
class TrackingFrameData:
    """Complete tracking data for one frame."""

    frame_index: int = 0
    timestamp_ms: float = 0.0

    # Face tracking
    face: FaceData = field(default_factory=FaceData)

    # Eye tracking
    left_eye: EyeData = field(default_factory=EyeData)
    right_eye: EyeData = field(default_factory=EyeData)

    # Blink tracking
    blink: BlinkData = field(default_factory=BlinkData)

    # Eye contact
    eye_contact: EyeContactData = field(default_factory=EyeContactData)

    # Overall state
    state: TrackingState = TrackingState.NO_FACE

    # Tracking quality
    overall_confidence: float = 0.0

    def __post_init__(self):
        self.overall_confidence = max(0.0, min(1.0, self.overall_confidence))

    def is_tracking(self) -> bool:
        """Check if currently tracking face/eyes."""
        return self.state in (
            TrackingState.TRACKING,
            TrackingState.EYE_CONTACT,
            TrackingState.BLINKING,
        )

    def has_eye_contact(self) -> bool:
        """Check if eye contact is established."""
        return self.state == TrackingState.EYE_CONTACT
