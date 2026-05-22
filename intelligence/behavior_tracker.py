"""Behavior tracker that distills tracking and activation snapshots into behavioral metrics."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

from intelligence.intelligence_data import BehaviorMetrics
from utils.logger import Logger


class BehaviorTracker:
    """
    Continuously observes tracking and activation data each frame and
    produces rolling ``BehaviorMetrics``.  Downstream systems (escalation,
    personality, director) consume these metrics without touching raw
    tracking data directly.
    """

    def __init__(self, window_s: float = 30.0, logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger(name="BehaviorTracker")
        self._window_s = max(5.0, window_s)
        self._start_time = time.perf_counter()

        # Rolling stare durations (ms)
        self._stare_durations: deque = deque(maxlen=200)
        self._current_stare_ms: float = 0.0
        self._was_staring: bool = False

        # Rolling blink timestamps
        self._blink_times: deque = deque(maxlen=200)
        self._was_blinking: bool = False
        self._no_blink_since: float = time.perf_counter()

        # Activation tracking
        self._activation_times: deque = deque(maxlen=100)
        self._activation_durations: deque = deque(maxlen=100)
        self._interruption_count: int = 0
        self._total_activations: int = 0

        # Face presence
        self._face_absent_since: Optional[float] = None
        self._face_absence_count: int = 0
        self._face_absence_total_ms: float = 0.0

        # Movement energy (face center delta)
        self._last_face_center: Optional[tuple] = None
        self._movement_samples: deque = deque(maxlen=60)

        # Output
        self.metrics = BehaviorMetrics()

        self.logger.info("BehaviorTracker initialized")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, tracking_data, activation_data) -> BehaviorMetrics:
        """
        Observe one frame of tracking + activation data and update metrics.

        Args:
            tracking_data: TrackingFrameData (or None)
            activation_data: ActivationSnapshot (or None)

        Returns:
            Updated ``BehaviorMetrics``.
        """
        now = time.perf_counter()
        session_s = now - self._start_time

        self._update_stare(tracking_data, activation_data, now)
        self._update_blinks(tracking_data, now)
        self._update_activations(activation_data, now)
        self._update_face_presence(tracking_data, now)
        self._update_movement(tracking_data)

        # Build output metrics
        m = self.metrics
        m.session_duration_s = session_s

        # Stare stats
        if self._stare_durations:
            m.avg_stare_duration_ms = sum(self._stare_durations) / len(
                self._stare_durations
            )
            m.max_stare_duration_ms = max(self._stare_durations)
        m.stare_count = len(self._stare_durations)

        # Blink rate
        window_start = now - self._window_s
        recent_blinks = [t for t in self._blink_times if t >= window_start]
        if self._window_s > 0:
            m.blink_rate_per_min = len(recent_blinks) / (self._window_s / 60.0)
        m.no_blink_streak_ms = (now - self._no_blink_since) * 1000.0

        # Activation stats
        m.activation_count = self._total_activations
        recent_acts = [t for t in self._activation_times if t >= window_start]
        if self._window_s > 0:
            m.activation_rate_per_min = len(recent_acts) / (self._window_s / 60.0)
        if self._activation_durations:
            m.avg_activation_duration_ms = sum(self._activation_durations) / len(
                self._activation_durations
            )
        m.interruption_count = self._interruption_count
        if self._total_activations + self._interruption_count > 0:
            m.interruption_rate = self._interruption_count / (
                self._total_activations + self._interruption_count
            )

        # Face presence
        m.face_absence_count = self._face_absence_count
        m.face_absence_total_ms = self._face_absence_total_ms

        # Movement energy
        if self._movement_samples:
            m.movement_energy = sum(self._movement_samples) / len(
                self._movement_samples
            )
        else:
            m.movement_energy = 0.0

        return m

    # ------------------------------------------------------------------
    # Internal observers
    # ------------------------------------------------------------------

    def _update_stare(self, tracking_data, activation_data, now: float) -> None:
        is_staring = False
        if activation_data is not None:
            is_staring = activation_data.state.value in (
                "STARE_DETECTED",
                "ACTIVATING",
                "POWER_ACTIVE",
            )

        if is_staring:
            if not self._was_staring:
                self._current_stare_ms = 0.0
            if activation_data is not None:
                self._current_stare_ms = max(
                    self._current_stare_ms, activation_data.stare_duration_ms
                )
        else:
            if self._was_staring and self._current_stare_ms > 0:
                self._stare_durations.append(self._current_stare_ms)
            self._current_stare_ms = 0.0

        self._was_staring = is_staring

    def _update_blinks(self, tracking_data, now: float) -> None:
        if tracking_data is None:
            return

        is_blinking = tracking_data.blink.is_blinking

        # Detect blink onset
        if is_blinking and not self._was_blinking:
            self._blink_times.append(now)
            self._no_blink_since = now

        if not is_blinking and not self._was_blinking:
            pass  # no-blink streak continues
        elif not is_blinking:
            self._no_blink_since = now

        self._was_blinking = is_blinking

    def _update_activations(self, activation_data, now: float) -> None:
        if activation_data is None:
            return

        state = activation_data.state.value

        if (
            state == "POWER_ACTIVE"
            and activation_data.activation_count > self._total_activations
        ):
            self._total_activations = activation_data.activation_count
            self._activation_times.append(now)
            if activation_data.activation_elapsed_ms > 0:
                self._activation_durations.append(activation_data.activation_elapsed_ms)

        if state == "INTERRUPTED":
            # Count unique interruptions (avoid double counting)
            expected = self._total_activations + self._interruption_count
            actual = activation_data.activation_count + (
                1 if state == "INTERRUPTED" else 0
            )
            if self._interruption_count < actual - activation_data.activation_count:
                self._interruption_count += 1

    def _update_face_presence(self, tracking_data, now: float) -> None:
        face_present = tracking_data is not None and tracking_data.face.detected

        if not face_present:
            if self._face_absent_since is None:
                self._face_absent_since = now
                self._face_absence_count += 1
        else:
            if self._face_absent_since is not None:
                absence_ms = (now - self._face_absent_since) * 1000.0
                self._face_absence_total_ms += absence_ms
                self._face_absent_since = None

    def _update_movement(self, tracking_data) -> None:
        if tracking_data is None or not tracking_data.face.detected:
            self._last_face_center = None
            return

        center = tracking_data.face.center
        if self._last_face_center is not None:
            dx = center[0] - self._last_face_center[0]
            dy = center[1] - self._last_face_center[1]
            energy = math.sqrt(dx * dx + dy * dy)
            self._movement_samples.append(energy)
        self._last_face_center = center

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._stare_durations.clear()
        self._blink_times.clear()
        self._activation_times.clear()
        self._activation_durations.clear()
        self._movement_samples.clear()
        self._interruption_count = 0
        self._total_activations = 0
        self._face_absence_count = 0
        self._face_absence_total_ms = 0.0
        self._start_time = time.perf_counter()
        self._no_blink_since = time.perf_counter()
        self.metrics = BehaviorMetrics()
