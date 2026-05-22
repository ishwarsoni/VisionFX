"""
High-level camera capture and frame handling module.
"""

import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import Logger


class CameraEngine:
    """
    Production-grade camera capture engine.
    Handles initialization, frame capture, and graceful error handling.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str = "default",
        logger: Optional[Logger] = None,
    ):
        """
        Initialize camera engine.

        Args:
            device_id: Camera device ID (usually 0 for default/built-in)
            width: Target frame width
            height: Target frame height
            fps: Target frames per second
            backend: Capture backend preference (default/auto/dshow/msmf/any)
            logger: Logger instance for diagnostic output
        """
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.backend = backend.lower().strip() if backend else "default"
        self.logger = logger or Logger(name="CameraEngine")

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self.frame_count = 0

        self._initialize()

    def _resolve_backend_candidates(self) -> List[Tuple[str, int]]:
        """Resolve backend candidates in preferred order for the current OS."""
        windows_backends = [
            ("msmf", cv2.CAP_MSMF),
            ("dshow", cv2.CAP_DSHOW),
            ("any", cv2.CAP_ANY),
        ]
        generic_backends = [("any", cv2.CAP_ANY)]

        if platform.system().lower() != "windows":
            return generic_backends

        if self.backend in {"default", "auto"}:
            return windows_backends

        by_name = {name: value for name, value in windows_backends}
        chosen = by_name.get(self.backend)
        if chosen is None:
            self.logger.warning(
                f"Unknown camera backend '{self.backend}', falling back to auto selection"
            )
            return windows_backends
        return [(self.backend, chosen)]

    def _resolve_device_candidates(self) -> List[int]:
        """Resolve camera indices to probe when primary index fails."""
        # Keep probe range narrow to avoid long startup while still handling remapped devices.
        nearby = [self.device_id, 0, 1, 2, 3]
        seen = set()
        ordered = []
        for idx in nearby:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered

    def _try_open_camera(self) -> Tuple[cv2.VideoCapture, int, str]:
        """Try opening camera across backend and device fallback candidates."""
        attempts = []
        backends = self._resolve_backend_candidates()
        devices = self._resolve_device_candidates()

        for device in devices:
            for backend_name, backend_flag in backends:
                cap = cv2.VideoCapture(device, backend_flag)
                if cap.isOpened():
                    self._apply_capture_properties(cap)
                    test_frame = self._read_probe_frame(cap)
                    if self._is_valid_probe_frame(test_frame):
                        self.logger.info(
                            f"Camera opened on device {device} using backend {backend_name.upper()}"
                        )
                        return cap, device, backend_name

                    reason = self._describe_invalid_probe_frame(test_frame)
                    self.logger.warning(
                        f"Rejected camera source device={device}, backend={backend_name.upper()} ({reason})"
                    )
                attempts.append(f"device={device},backend={backend_name}")
                cap.release()

        attempted = ", ".join(attempts)
        raise RuntimeError(
            f"Failed to open camera. Tried: {attempted}. "
            "Close apps using the camera (Zoom/Teams/Browser) and retry."
        )

    def _apply_capture_properties(self, cap: cv2.VideoCapture) -> None:
        """Apply target capture properties to an open capture device."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time

    def _read_probe_frame(
        self, cap: cv2.VideoCapture, max_reads: int = 6
    ) -> Optional[np.ndarray]:
        """Read a probe frame with a few attempts to allow backend warmup."""
        frame = None
        for _ in range(max_reads):
            try:
                ret, candidate = cap.read()
            except cv2.error:
                continue
            if ret and candidate is not None and candidate.size > 0:
                frame = candidate
        return frame

    def _safe_read_frame(
        self, cap: cv2.VideoCapture, max_reads: int = 3
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame while guarding against intermittent backend read exceptions."""
        for _ in range(max_reads):
            try:
                ret, frame = cap.read()
            except cv2.error:
                continue
            if ret and frame is not None and frame.size > 0:
                return True, frame
        return False, None

    @staticmethod
    def _is_valid_probe_frame(frame: Optional[np.ndarray]) -> bool:
        """Detect obviously broken camera feeds (solid color / empty frames)."""
        if frame is None or frame.size == 0:
            return False
        if len(frame.shape) < 2:
            return False

        if len(frame.shape) == 3:
            per_channel_std = np.std(frame.reshape(-1, frame.shape[2]), axis=0)
            max_std = float(np.max(per_channel_std))
            return max_std >= 1.0

        return float(np.std(frame)) >= 1.0

    @staticmethod
    def _describe_invalid_probe_frame(frame: Optional[np.ndarray]) -> str:
        """Human-readable reason for an invalid probe frame."""
        if frame is None:
            return "no frame"
        if frame.size == 0:
            return "empty frame"
        if len(frame.shape) == 3:
            mean = tuple(float(np.mean(frame[:, :, i])) for i in range(frame.shape[2]))
            std = tuple(float(np.std(frame[:, :, i])) for i in range(frame.shape[2]))
            return f"uniform frame mean={mean} std={std}"
        return f"uniform frame mean={float(np.mean(frame)):.2f} std={float(np.std(frame)):.2f}"

    def _initialize(self) -> None:
        """Initialize camera capture with proper error handling."""
        try:
            self.cap, opened_device, opened_backend = self._try_open_camera()
            self.device_id = opened_device
            self.backend = opened_backend

            # Read a test frame to verify
            ret, frame = self._safe_read_frame(self.cap)
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from camera")

            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.is_open = True
            self.logger.info(f"Camera initialized successfully")
            self.logger.info(
                f"Capture: device={self.device_id}, backend={self.backend.upper()}"
            )
            self.logger.info(
                f"Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
            )

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.is_open = False
            if self.cap:
                self.cap.release()
            raise

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from camera.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if not self.is_open or self.cap is None:
            return False, None

        try:
            ret, frame = self._safe_read_frame(self.cap)
            if ret and frame is not None:
                self.frame_count += 1
                return True, frame
            return False, None
        except Exception as e:
            self.logger.error(f"Frame read error: {e}")
            return False, None

    def get_frame_info(self) -> dict:
        """Get current camera frame information."""
        if not self.is_open or self.cap is None:
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "frame_count": self.frame_count,
        }

    def release(self) -> None:
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.is_open = False
            self.logger.info("Camera released")

    def __del__(self):
        """Ensure camera is released on object destruction."""
        self.release()
