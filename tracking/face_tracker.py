"""
Face tracking using MediaPipe FaceMesh.
Detects and tracks face landmarks in real-time using Tasks API.
"""

import io
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.tracking_data import FaceData
from utils.logger import Logger


class FaceTracker:
    """
    Production-grade face tracking using MediaPipe FaceLandmarker.
    Provides stable face detection and 478 landmark points.
    """

    FACE_OVALS_LANDMARK_INDICES = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
        10,
    ]

    LEFT_EYE_INDICES = list(range(33, 48))
    RIGHT_EYE_INDICES = list(range(263, 278))

    def __init__(
        self,
        logger: Optional[Logger] = None,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
    ):
        self.logger = logger or Logger(name="FaceTracker")
        self.max_num_faces = max(1, max_num_faces)
        self._face_detector = None
        self._mediapipe_available = False
        self.frame_count = 0
        self.last_eyes_cache = None

        self._init_media_pipe()

        if not self._mediapipe_available:
            self._init_haar_fallback()

    def _init_media_pipe(self):
        """Initialize MediaPipe FaceLandmarker using Tasks API."""
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

            with self._suppress_external_logs():
                from mediapipe.tasks.python import BaseOptions, vision

            model_path = os.path.expanduser("~/.mediapipe/models/face_landmarker.task")

            if not os.path.exists(model_path):
                self.logger.info("Downloading FaceLandmarker model...")
                import urllib.request

                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.max_num_faces,
                output_face_blendshapes=False,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            with self._suppress_external_logs():
                self._face_detector = vision.FaceLandmarker.create_from_options(options)
            self._mediapipe_available = True
            self.is_initialized = True

        except Exception as e:
            self._mediapipe_available = False
            self._face_detector = None

    def _on_face_result(self, result, output_image, timestamp_ms):
        """Callback for face detection results."""
        pass

    def _init_haar_fallback(self):
        """Initialize Haar Cascade fallback."""
        self.logger.info("Initializing Haar Cascade fallback")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.fallback_classifier = cv2.CascadeClassifier(cascade_path)

        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_classifier = cv2.CascadeClassifier(eye_cascade_path)

        self.is_initialized = (
            not self.fallback_classifier.empty() and not self.eye_classifier.empty()
        )

        if not self.is_initialized:
            self.logger.error("Failed to load Haar cascades!")

    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """Detect face and landmarks in frame."""
        if not self.is_initialized:
            return None

        try:
            self.frame_count += 1

            if self._mediapipe_available and self._face_detector:
                return self._detect_media_pipe(frame)
            else:
                return self._detect_haar(frame)

        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return FaceData(detected=False)

    def _detect_media_pipe(self, frame: np.ndarray) -> FaceData:
        """Detect face using MediaPipe Tasks API."""
        with self._suppress_external_logs():
            import mediapipe as mp
            from mediapipe.tasks.python import vision

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        with self._suppress_external_logs():
            results = self._face_detector.detect_for_video(
                mp_image, int(self.frame_count * 33)
            )

        if (
            not results
            or not results.face_landmarks
            or len(results.face_landmarks) == 0
        ):
            return FaceData(detected=False)

        face_landmarks = results.face_landmarks[0]
        h, w, _ = frame.shape

        landmarks = []
        for landmark in face_landmarks:
            landmarks.append((int(landmark.x * w), int(landmark.y * h)))

        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        padding = int((x_max - x_min) * 0.1)
        bbox = (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(w, x_max + padding) - max(0, x_min - padding),
            min(h, y_max + padding) - max(0, y_min - padding),
        )

        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        rotation = self._estimate_rotation(landmarks[1], landmarks[168])

        return FaceData(
            detected=True,
            center=(center_x, center_y),
            bounding_box=bbox,
            landmarks=landmarks,
            confidence=0.85,
            rotation_degrees=rotation,
        )

    def _detect_haar(self, frame: np.ndarray) -> FaceData:
        """Detect face using Haar Cascade fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.fallback_classifier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        if len(faces) == 0:
            return FaceData(detected=False)

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        roi_gray = gray[y : int(y + fh * 0.6), x : x + fw]
        eyes = self.eye_classifier.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[-1]

            new_lex = x + ex1 + ew1 // 2
            new_ley = y + ey1 + eh1 // 2
            new_rex = x + ex2 + ew2 // 2
            new_rey = y + ey2 + eh2 // 2
            new_eye_w = max(ew1, ew2)

            if self.last_eyes_cache:
                _, _, old_lex, old_ley, old_rex, old_rey, old_eye_w = (
                    self.last_eyes_cache
                )
                alpha = 0.15
                lex = int(old_lex * (1 - alpha) + new_lex * alpha)
                ley = int(old_ley * (1 - alpha) + new_ley * alpha)
                rex = int(old_rex * (1 - alpha) + new_rex * alpha)
                rey = int(old_rey * (1 - alpha) + new_rey * alpha)
                eye_w = int(old_eye_w * (1 - alpha) + new_eye_w * alpha)
            else:
                lex, ley, rex, rey, eye_w = (
                    new_lex,
                    new_ley,
                    new_rex,
                    new_rey,
                    new_eye_w,
                )

            self.last_eyes_cache = (None, None, lex, ley, rex, rey, eye_w)

        landmarks = [(0, 0)] * 478

        if self.last_eyes_cache:
            _, _, lex, ley, rex, rey, eye_w = self.last_eyes_cache
        else:
            lex, ley = int(x + fw * 0.35), int(y + fh * 0.35)
            rex, rey = int(x + fw * 0.65), int(y + fh * 0.35)
            eye_w = int(fw * 0.2)

        landmarks[1] = (int(x + fw * 0.5), int(y + fh * 0.55))
        landmarks[168] = (int(x + fw * 0.5), int(y + fh * 0.35))

        for i in self.LEFT_EYE_INDICES:
            landmarks[i] = (lex, ley)
        half_ew = eye_w // 2
        landmarks[33] = (lex - half_ew, ley)
        landmarks[133] = (lex + half_ew, ley)
        landmarks[159] = (lex, ley - 5)
        landmarks[145] = (lex, ley + 5)

        for i in self.RIGHT_EYE_INDICES:
            landmarks[i] = (rex, rey)
        landmarks[362] = (rex + half_ew, rey)
        landmarks[263] = (rex - half_ew, rey)
        landmarks[386] = (rex, rey - 5)
        landmarks[374] = (rex, rey + 5)

        landmarks[10] = (int(x + fw * 0.5), y)
        landmarks[152] = (int(x + fw * 0.5), y + fh)
        landmarks[234] = (x, int(y + fh * 0.5))
        landmarks[454] = (x + fw, int(y + fh * 0.5))

        return FaceData(
            detected=True,
            center=(x + fw / 2, y + fh / 2),
            bounding_box=(x, y, fw, fh),
            landmarks=landmarks,
            confidence=0.8,
            rotation_degrees=0.0,
        )

    def _estimate_rotation(
        self, nose_tip: Tuple[int, int], nose_bridge: Tuple[int, int]
    ) -> float:
        """Estimate head rotation using nose direction."""
        import math

        dy = nose_bridge[1] - nose_tip[1]
        dx = nose_bridge[0] - nose_tip[0]
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def get_eye_region_landmarks(
        self, face_data: FaceData, eye: str = "left"
    ) -> List[Tuple[int, int]]:
        """Get landmarks for eye region."""
        indices = self.LEFT_EYE_INDICES if eye == "left" else self.RIGHT_EYE_INDICES
        return [face_data.landmarks[i] for i in indices if i < len(face_data.landmarks)]

    def release(self) -> None:
        """Release resources."""
        if self._face_detector:
            self._face_detector.close()
            self.is_initialized = False
            self.logger.info("FaceTracker released")

    @staticmethod
    @contextmanager
    def _suppress_external_logs():
        with open(os.devnull, "w") as devnull:
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    yield
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

    def __del__(self):
        """Ensure cleanup."""
        self.release()
