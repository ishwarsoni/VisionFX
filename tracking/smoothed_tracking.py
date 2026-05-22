"""
Kalman-filtered eye tracking for rock-stable VFX compositing.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class KalmanFilter2D:
    """2D Kalman filter for smooth tracking."""

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = (
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            * process_noise
        )
        self.kalman.measurementNoiseCov = (
            np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
        )
        self._initialized = False

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[x], [y]], np.float32)
        if not self._initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self._initialized = True
            return (x, y)
        prediction = self.kalman.predict()
        self.kalman.correct(measurement)
        return (float(prediction[0]), float(prediction[1]))

    def reset(self):
        self._initialized = False


class EMASmoother:
    """Exponential moving average for temporal smoothing."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._last = None

    def update(self, value: float) -> float:
        if self._last is None:
            self._last = value
            return value
        smoothed = self.alpha * value + (1 - self.alpha) * self._last
        self._last = smoothed
        return smoothed

    def update_point(self, x: float, y: float) -> Tuple[float, float]:
        return (self.update(x), self.update(y))

    def reset(self):
        self._last = None


class OneEuroFilter:
    """One Euro Filter for velocity-adaptive natural motion."""

    def __init__(
        self, min_cutoff: float = 0.5, beta: float = 0.7, d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._last_value = None
        self._last_derivative = 0.0
        self._initialized = False

    def update(self, value: float, dt: float = 1 / 30) -> float:
        if not self._initialized:
            self._last_value = value
            self._initialized = True
            return value

        speed = abs(value - self._last_value) / max(dt, 0.001)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self.d_cutoff / (self.d_cutoff + cutoff)

        filtered = alpha * value + (1 - alpha) * self._last_value
        self._last_value = filtered
        return filtered

    def update_point(
        self, x: float, y: float, dt: float = 1 / 30
    ) -> Tuple[float, float]:
        return (self.update(x, dt), self.update(y, dt))

    def reset(self):
        self._last_value = None
        self._initialized = False


class MicroMotion:
    """Natural micro-drift for believable eye movement."""

    def __init__(self, intensity: float = 1.5):
        self.intensity = intensity
        self._phase_x = np.random.rand() * 2 * np.pi
        self._phase_y = np.random.rand() * 2 * np.pi
        self._phase_beat = np.random.rand() * 2 * np.pi
        self._last_saccade = 0.0
        self._saccade_target = (0.0, 0.0)

    def add_drift(self, x: float, y: float, time: float) -> Tuple[float, float]:
        breath = np.sin(time * 0.8 + self._phase_beat) * 0.3 + 0.7

        drift_x = np.sin(time * 0.4 + self._phase_x) * self.intensity * breath
        drift_y = np.cos(time * 0.25 + self._phase_y) * self.intensity * breath * 0.7

        drift_x += np.sin(time * 1.2 + self._phase_beat) * 0.3
        drift_y += np.cos(time * 0.9) * 0.2

        if time - self._last_saccade > 3.0 and np.random.rand() < 0.008:
            self._saccade_target = (np.random.uniform(-2, 2), np.random.uniform(-1, 1))
            self._last_saccade = time

        current_saccade = self._saccade_target[0] * (
            1.0 - min(1.0, (time - self._last_saccade) / 0.5)
        )
        current_saccade_y = self._saccade_target[1] * (
            1.0 - min(1.0, (time - self._last_saccade) / 0.5)
        )

        return x + drift_x + current_saccade, y + drift_y + current_saccade_y

    def add_saccade(self, x: float, y: float) -> Tuple[float, float]:
        return x, y


class EyeTrackerStabilizer:
    """Complete stabilization system with natural motion."""

    def __init__(self):
        self._iris_left = KalmanFilter2D(process_noise=3e-4, measurement_noise=2e-2)
        self._iris_right = KalmanFilter2D(process_noise=3e-4, measurement_noise=2e-2)
        self._center_left = KalmanFilter2D(process_noise=3e-4, measurement_noise=2e-2)
        self._center_right = KalmanFilter2D(process_noise=3e-4, measurement_noise=2e-2)

        self._angle_left = OneEuroFilter(min_cutoff=0.2, beta=0.3, d_cutoff=1.0)
        self._angle_right = OneEuroFilter(min_cutoff=0.2, beta=0.3, d_cutoff=1.0)
        self._tilt_left = OneEuroFilter(min_cutoff=0.2, beta=0.3, d_cutoff=1.0)
        self._tilt_right = OneEuroFilter(min_cutoff=0.2, beta=0.3, d_cutoff=1.0)
        self._width_left = OneEuroFilter(min_cutoff=0.3, beta=0.2, d_cutoff=1.0)
        self._width_right = OneEuroFilter(min_cutoff=0.3, beta=0.2, d_cutoff=1.0)

        self._confidence_filter = EMASmoother(0.15)
        self._last_confidence = 0.5

        self._micro_motion = MicroMotion(intensity=0.5)
        self._time = 0.0

    def set_time(self, time: float):
        self._time = time

    def stabilize(
        self,
        iris_left: Tuple,
        iris_right: Tuple,
        center_left: Tuple,
        center_right: Tuple,
        angle_left: float,
        angle_right: float,
        tilt_left: float,
        tilt_right: float,
        width_left: float,
        width_right: float,
        confidence: float,
    ) -> dict:

        stable_iris_left = self._iris_left.update(iris_left[0], iris_left[1])
        stable_iris_right = self._iris_right.update(iris_right[0], iris_right[1])

        stable_center_left = self._center_left.update(center_left[0], center_left[1])
        stable_center_right = self._center_right.update(
            center_right[0], center_right[1]
        )

        stable_angle_left = self._angle_left.update(angle_left)
        stable_angle_right = self._angle_right.update(angle_right)
        stable_tilt_left = self._tilt_left.update(tilt_left)
        stable_tilt_right = self._tilt_right.update(tilt_right)
        stable_width_left = self._width_left.update(width_left)
        stable_width_right = self._width_right.update(width_right)

        stable_confidence = self._confidence_filter.update(confidence)

        stable_iris_left = self._micro_motion.add_drift(
            stable_iris_left[0], stable_iris_left[1], self._time
        )
        stable_iris_right = self._micro_motion.add_drift(
            stable_iris_right[0], stable_iris_right[1], self._time
        )
        stable_iris_left = self._micro_motion.add_saccade(
            stable_iris_left[0], stable_iris_left[1]
        )
        stable_iris_right = self._micro_motion.add_saccade(
            stable_iris_right[0], stable_iris_right[1]
        )

        return {
            "iris_left": stable_iris_left,
            "iris_right": stable_iris_right,
            "center_left": stable_center_left,
            "center_right": stable_center_right,
            "angle_left": stable_angle_left,
            "angle_right": stable_angle_right,
            "tilt_left": stable_tilt_left,
            "tilt_right": stable_tilt_right,
            "width_left": stable_width_left,
            "width_right": stable_width_right,
            "confidence": stable_confidence,
        }

    def reset(self):
        self._iris_left.reset()
        self._iris_right.reset()
        self._center_left.reset()
        self._center_right.reset()
        self._angle_left.reset()
        self._angle_right.reset()
        self._tilt_left.reset()
        self._tilt_right.reset()
        self._width_left.reset()
        self._width_right.reset()
        self._confidence_filter.reset()
