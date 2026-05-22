"""
Frame processing utilities for common operations.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class FrameProcessor:
    """
    Utility class for common frame processing operations.
    Provides optimized, reusable frame transformation functions.
    """

    @staticmethod
    def resize_frame(
        frame: np.ndarray,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize frame to target dimensions.

        Args:
            frame: Input frame
            width: Target width
            height: Target height
            interpolation: Interpolation method

        Returns:
            Resized frame
        """
        return cv2.resize(frame, (width, height), interpolation=interpolation)

    @staticmethod
    def flip_horizontal(frame: np.ndarray) -> np.ndarray:
        """
        Flip frame horizontally (mirror effect).

        Args:
            frame: Input frame

        Returns:
            Horizontally flipped frame
        """
        return cv2.flip(frame, 1)

    @staticmethod
    def convert_color(
        frame: np.ndarray, from_space: str = "BGR", to_space: str = "RGB"
    ) -> np.ndarray:
        """
        Convert frame between color spaces.

        Args:
            frame: Input frame
            from_space: Source color space (BGR, RGB, HSV, GRAY, etc.)
            to_space: Target color space

        Returns:
            Converted frame
        """
        color_conversions = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
            ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
            ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
        }

        key = (from_space, to_space)
        if key not in color_conversions:
            raise ValueError(f"Unsupported conversion: {from_space} → {to_space}")

        return cv2.cvtColor(frame, color_conversions[key])

    @staticmethod
    def add_padding(
        frame: np.ndarray, padding: int, color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Add padding around frame.

        Args:
            frame: Input frame
            padding: Padding size in pixels
            color: Padding color (BGR)

        Returns:
            Padded frame
        """
        return cv2.copyMakeBorder(
            frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=color
        )

    @staticmethod
    def get_frame_info(frame: np.ndarray) -> dict:
        """
        Get information about frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary with frame info
        """
        h, w = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        dtype = str(frame.dtype)

        return {
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": dtype,
            "size_bytes": frame.nbytes,
        }

    @staticmethod
    def optimize_frame(
        frame: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        flip: bool = False,
        grayscale: bool = False,
    ) -> np.ndarray:
        """
        Apply common optimization pipeline to frame.

        Args:
            frame: Input frame
            target_width: Target width (None to keep original)
            target_height: Target height (None to keep original)
            flip: Whether to flip horizontally
            grayscale: Whether to convert to grayscale

        Returns:
            Processed frame
        """
        result = frame.copy()

        if flip:
            result = FrameProcessor.flip_horizontal(result)

        if grayscale:
            result = FrameProcessor.convert_color(result, "BGR", "GRAY")

        if target_width and target_height:
            result = FrameProcessor.resize_frame(result, target_width, target_height)

        return result

    @staticmethod
    def is_nearly_black(
        frame: np.ndarray,
        brightness_threshold: float = 8.0,
        coverage_threshold: float = 0.985,
    ) -> bool:
        """Detect whether a frame has been collapsed to an effectively black image."""
        if frame is None or frame.size == 0:
            return True

        gray = frame
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_brightness = float(np.mean(gray))
        dark_pixel_ratio = float(np.mean(gray < brightness_threshold))
        return (
            mean_brightness < brightness_threshold
            or dark_pixel_ratio >= coverage_threshold
        )

    @staticmethod
    def ensure_visible_frame(frame: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Return the fallback frame if the processed frame has collapsed to black."""
        if FrameProcessor.is_nearly_black(frame):
            return fallback.copy()
        return frame

    @staticmethod
    def enhance_visibility(frame: np.ndarray, target_mean: float = 48.0) -> np.ndarray:
        """Lift a very dark camera frame into a usable exposure range."""
        if frame is None or frame.size == 0:
            return frame

        result = frame.copy()
        current_mean = float(np.mean(result))

        if len(result.shape) == 3:
            ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            y = clahe.apply(y)

            brightness_gain = 1.0
            if current_mean < target_mean:
                brightness_gain = min(
                    3.0, max(1.0, target_mean / max(1.0, current_mean))
                )

            y = cv2.convertScaleAbs(y, alpha=brightness_gain, beta=4)
            ycrcb = cv2.merge((y, cr, cb))
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            if current_mean < target_mean:
                gain = min(3.0, max(1.0, target_mean / max(1.0, current_mean)))
                result = cv2.convertScaleAbs(result, alpha=gain, beta=4)

        blurred = cv2.GaussianBlur(result, (0, 0), 0.8)
        result = cv2.addWeighted(result, 1.04, blurred, -0.04, 0)

        return result
