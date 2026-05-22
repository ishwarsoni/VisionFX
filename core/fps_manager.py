"""
Real-time FPS tracking and management module.
"""

import time
from collections import deque
from typing import Optional

from utils.logger import Logger


class FPSManager:
    """
    Production-grade FPS tracking with exponential smoothing.
    Provides real-time FPS monitoring and optional frame rate limiting.
    """

    def __init__(
        self,
        smoothing_window: int = 30,
        fps_cap: Optional[int] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize FPS manager.

        Args:
            smoothing_window: Number of frames for FPS smoothing
            fps_cap: Optional maximum FPS to enforce
            logger: Logger instance
        """
        self.smoothing_window = smoothing_window
        self.fps_cap = fps_cap
        self.frame_times = deque(maxlen=smoothing_window)
        self.logger = logger or Logger(name="FPSManager")

        self.current_fps = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0

        if fps_cap:
            self.frame_interval = 1.0 / fps_cap
            self.logger.info(f"FPS capped at {fps_cap}")
        else:
            self.frame_interval = 0.0

    def tick(self) -> float:
        """
        Mark a frame tick and return current FPS.
        Should be called once per frame in main loop.

        Returns:
            Current smoothed FPS value
        """
        current_time = time.time()
        delta_time = current_time - self.last_frame_time

        self.frame_times.append(delta_time)
        self.frame_count += 1

        # Calculate smoothed FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                self.current_fps = 1.0 / avg_frame_time

        self.last_frame_time = current_time
        return self.current_fps

    def wait_for_frame_interval(self) -> None:
        """
        Apply frame rate limiting by sleeping if necessary.
        Should be called at the end of frame processing loop.
        """
        if self.fps_cap is None:
            return

        elapsed = time.time() - self.last_frame_time
        sleep_time = self.frame_interval - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    def get_fps(self) -> float:
        """Get current smoothed FPS value."""
        return self.current_fps

    def get_frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if self.current_fps > 0:
            return 1000.0 / self.current_fps
        return 0.0

    def get_stats(self) -> dict:
        """Get detailed FPS statistics."""
        if len(self.frame_times) == 0:
            return {"fps": 0.0, "frame_time_ms": 0.0, "frame_count": 0}

        frame_times_list = list(self.frame_times)
        avg_time = sum(frame_times_list) / len(frame_times_list)

        return {
            "fps": self.current_fps,
            "frame_time_ms": avg_time * 1000,
            "frame_count": self.frame_count,
            "samples": len(self.frame_times),
            "fps_cap": self.fps_cap,
        }

    def reset(self) -> None:
        """Reset FPS tracking."""
        self.frame_times.clear()
        self.current_fps = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()
