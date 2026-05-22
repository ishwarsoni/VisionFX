"""
Webcam Quality Enhancement
Subtle preprocessing for cleaner, more professional compositing.
"""

import cv2
import numpy as np


class WebcamEnhancer:
    """
    Cinematic webcam preprocessing for professional quality.
    Subtle enhancements - avoids AI filter look.
    """

    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self._last_frame = None
        self._denoise_cache = None

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle cinematic enhancement to webcam feed."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        if h < 100 or w < 100:
            return frame

        result = frame.astype(np.float32)

        bright = np.mean(result)
        if bright < 80:
            gain = 1.0 + (100 - bright) / 400
            result = result * gain
            result = np.clip(result, 0, 255)

        luma = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2YUV)[:, :, 0]
        luma = self._clahe.apply(luma)

        yuv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = luma
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.float32)

        result = self._subtle_sharpen(result)

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _subtle_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Very subtle sharpening - avoids plastic look."""
        kernel = (
            np.array(
                [[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]],
                dtype=np.float32,
            )
            / 3.0
        )

        sharpened = cv2.filter2D(img, -1, kernel)

        return img * 0.85 + sharpened * 0.15

    def denoise_light(self, frame: np.ndarray) -> np.ndarray:
        """Light denoising - preserve skin detail."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            return frame

        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 2, 2, 7, 21)

        return denoised

    def apply_cinematic_tone(self, frame: np.ndarray) -> np.ndarray:
        """Subtle cinematic tone mapping."""
        if frame is None or frame.size == 0:
            return frame

        result = frame.astype(np.float32)

        result = result * 0.98

        lift = 3
        result = result + lift

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


def create_webcam_enhancer() -> WebcamEnhancer:
    """Factory function."""
    return WebcamEnhancer()
