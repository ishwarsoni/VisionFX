"""Organized export pipeline and directory management."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from recording.recording_data import ExportInfo
from utils.logger import Logger


class ExportManager:
    """
    Manages the recordings directory tree and generates timestamped
    filenames for sessions, replays, and screenshots.
    """

    def __init__(
        self,
        export_root: str = "recordings",
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="ExportManager")
        self.export_root = export_root

        # Sub-directories
        self.sessions_dir = os.path.join(export_root, "sessions")
        self.replays_dir = os.path.join(export_root, "replays")
        self.screenshots_dir = os.path.join(export_root, "screenshots")
        self.exports_dir = os.path.join(export_root, "exports")

        self._ensure_directories()
        self.logger.info(f"ExportManager initialized: {self.export_root}/")

    def _ensure_directories(self) -> None:
        for d in (
            self.sessions_dir,
            self.replays_dir,
            self.screenshots_dir,
            self.exports_dir,
        ):
            os.makedirs(d, exist_ok=True)

    def get_session_path(self, extension: str = "mp4") -> str:
        ext = extension.strip(".")
        filename = f"session_{self._timestamp()}.{ext}"
        return os.path.join(self.sessions_dir, filename)

    def get_replay_path(self, extension: str = "mp4") -> str:
        ext = extension.strip(".")
        filename = f"replay_{self._timestamp()}.{ext}"
        return os.path.join(self.replays_dir, filename)

    def get_screenshot_path(self, extension: str = "png") -> str:
        ext = extension.strip(".")
        filename = f"sharingan_{self._timestamp()}.{ext}"
        return os.path.join(self.screenshots_dir, filename)

    def file_info(self, path: str) -> ExportInfo:
        if not os.path.exists(path):
            return ExportInfo(path=path, success=False, error="File not found")
        return ExportInfo(
            path=path,
            file_size_bytes=os.path.getsize(path),
            success=True,
        )

    @staticmethod
    def _timestamp() -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
