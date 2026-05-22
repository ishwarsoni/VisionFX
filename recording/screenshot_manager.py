"""Cinematic screenshot capture with timestamped filenames."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from recording.recording_data import ExportInfo
from utils.logger import Logger


class ScreenshotManager:
    """
    Captures the current composited frame and writes it to disk as a
    high-quality PNG (default) or JPEG image.
    """

    def __init__(
        self,
        output_dir: str = "recordings/screenshots",
        quality: int = 95,
        image_format: str = "png",
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="ScreenshotManager")
        self.output_dir = output_dir
        self.quality = max(0, min(100, quality))
        self.image_format = image_format.lower().strip(".")
        self.screenshot_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(
            f"ScreenshotManager initialized: {self.output_dir} "
            f"(format={self.image_format}, quality={self.quality})"
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        frame: np.ndarray,
        output_dir: Optional[str] = None,
    ) -> ExportInfo:
        """
        Save the current composited frame to disk.

        Args:
            frame: Final composited BGR frame to save.
            output_dir: Override output directory (optional).

        Returns:
            ``ExportInfo`` describing the saved file.
        """
        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        filename = self._generate_filename()
        path = os.path.join(target_dir, filename)

        try:
            if self.image_format == "jpg" or self.image_format == "jpeg":
                params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            else:
                # PNG compression (0-9, lower = faster)
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            success = cv2.imwrite(path, frame, params)

            if not success:
                self.logger.error(f"Screenshot write failed: {path}")
                return ExportInfo(
                    path=path, success=False, error="cv2.imwrite returned False"
                )

            self.screenshot_count += 1
            h, w = frame.shape[:2]
            file_size = os.path.getsize(path) if os.path.exists(path) else 0

            info = ExportInfo(
                path=path,
                format=self.image_format,
                width=w,
                height=h,
                file_size_bytes=file_size,
                frame_count=1,
                success=True,
            )

            self.logger.info(
                f"Screenshot saved: {filename} ({w}x{h}, {file_size / 1024:.1f}KB)"
            )
            return info

        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return ExportInfo(path=path, success=False, error=str(e))

    # ------------------------------------------------------------------
    # Filename generation
    # ------------------------------------------------------------------

    def _generate_filename(self) -> str:
        """Generate a timestamped filename."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        millis = f"{now.microsecond // 1000:03d}"
        return f"sharingan_{timestamp}_{millis}.{self.image_format}"
