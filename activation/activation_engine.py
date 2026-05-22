"""Phase 3 activation intelligence engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from activation.activation_data import ActivationSnapshot
from activation.activation_state import ActivationState
from activation.activation_timer import ActivationTimer
from activation.cooldown_manager import CooldownManager
from activation.event_dispatcher import ActivationEventDispatcher
from activation.state_machine import ActivationStateMachine
from activation.transition_manager import ActivationTransitionManager
from config.settings import ActivationConfig
from tracking.tracking_data import TrackingFrameData
from utils.logger import Logger


class ActivationEngine:
    """Central activation controller for eye power buildup and lockout."""

    def __init__(self, config: ActivationConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger(name="ActivationEngine")

        self.state_machine = ActivationStateMachine(
            transition_cooldown_ms=config.transition_stability_ms
        )
        self.transition_manager = ActivationTransitionManager(
            stability_window_ms=config.transition_stability_ms,
            interruption_threshold_ms=config.interruption_threshold_ms,
        )
        self.cooldown_manager = CooldownManager()
        self.event_dispatcher = ActivationEventDispatcher()

        self.stare_timer = ActivationTimer()
        self.activation_timer = ActivationTimer()
        self.interruption_timer = ActivationTimer()

        self.snapshot = ActivationSnapshot()
        self.last_update_time_ms: Optional[float] = None
        self.activation_count = 0

        self.logger.info("ActivationEngine initialized successfully")

    def on(self, event_name: str, handler) -> None:
        self.event_dispatcher.on(event_name, handler)

    def off(self, event_name: str, handler) -> None:
        self.event_dispatcher.off(event_name, handler)

    def update(self, tracking_data: Optional[TrackingFrameData]) -> ActivationSnapshot:
        now_ms = time.perf_counter() * 1000.0
        delta_ms = (
            0.0
            if self.last_update_time_ms is None
            else max(0.0, now_ms - self.last_update_time_ms)
        )
        self.last_update_time_ms = now_ms

        self._advance_timers(delta_ms)
        self._ingest_tracking_data(tracking_data)
        self._process_cooldown(now_ms, tracking_data)
        self._process_interruptions(now_ms, delta_ms, tracking_data)
        self._process_baseline_transitions(now_ms, delta_ms, tracking_data)
        self._process_activation_progress(now_ms)
        self._refresh_snapshot(tracking_data)
        return self.snapshot

    def reset(self) -> None:
        self.state_machine.reset()
        self.transition_manager.reset()
        self.cooldown_manager.reset()
        self.stare_timer.reset()
        self.activation_timer.reset()
        self.interruption_timer.reset()
        self.snapshot = ActivationSnapshot()
        self.last_update_time_ms = None
        self.activation_count = 0

    def get_debug_lines(self) -> List[str]:
        return self.snapshot.debug_lines()

    def _advance_timers(self, delta_ms: float) -> None:
        self.stare_timer.advance(delta_ms)
        self.activation_timer.advance(delta_ms)
        self.interruption_timer.advance(delta_ms)
        self.cooldown_manager.advance(delta_ms)

    def _ingest_tracking_data(self, tracking_data: Optional[TrackingFrameData]) -> None:
        if tracking_data is None:
            self.snapshot.face_detected = False
            self.snapshot.eye_contact = False
            self.snapshot.blink_detected = False
            self.snapshot.tracking_confidence = 0.0
            return

        self.snapshot.face_detected = tracking_data.face.detected
        self.snapshot.eye_contact = tracking_data.eye_contact.in_contact
        self.snapshot.blink_detected = tracking_data.blink.is_blinking
        self.snapshot.tracking_confidence = max(
            0.0, min(1.0, tracking_data.overall_confidence)
        )

    def _process_cooldown(
        self, now_ms: float, tracking_data: Optional[TrackingFrameData]
    ) -> None:
        if (
            self.cooldown_manager.timer.running
            and self.state_machine.current_state != ActivationState.COOLDOWN
        ):
            if self.state_machine.transition(
                ActivationState.COOLDOWN, now_ms, "cooldown_started", force=True
            ):
                self.event_dispatcher.emit(
                    "on_cooldown_start", self._build_payload(reason="cooldown_started")
                )

        if not self.cooldown_manager.timer.running:
            return

        if self.cooldown_manager.is_complete:
            self.cooldown_manager.stop()
            self.event_dispatcher.emit(
                "on_cooldown_end", self._build_payload(reason="cooldown_complete")
            )
            next_state = self._derive_baseline_state(tracking_data)
            self.state_machine.transition(
                next_state, now_ms, "cooldown_complete", force=True
            )

    def _process_interruptions(
        self, now_ms: float, delta_ms: float, tracking_data: Optional[TrackingFrameData]
    ) -> None:
        interrupt_reason = self._get_interrupt_reason(tracking_data)
        interrupt_candidate = interrupt_reason is not None

        if interrupt_candidate:
            self.interruption_timer.start(reset=False)
            self.interruption_timer.advance(delta_ms)
        else:
            self.interruption_timer.reset()

        if not self.transition_manager.update_interrupt(
            interrupt_candidate, delta_ms, interrupt_reason or ""
        ):
            return

        if self.state_machine.current_state in (
            ActivationState.ACTIVATING,
            ActivationState.STARE_DETECTED,
            ActivationState.POWER_ACTIVE,
        ):
            self._enter_interrupted(now_ms, interrupt_reason or "interrupt")

    def _process_baseline_transitions(
        self, now_ms: float, delta_ms: float, tracking_data: Optional[TrackingFrameData]
    ) -> None:
        baseline_state = self._derive_baseline_state(tracking_data)
        face_ready = baseline_state in (
            ActivationState.FACE_DETECTED,
            ActivationState.TRACKING,
        )

        if not self.transition_manager.update_face(face_ready, delta_ms):
            if self.state_machine.current_state == ActivationState.IDLE and face_ready:
                self.state_machine.transition(
                    ActivationState.FACE_DETECTED, now_ms, "face_detected", force=True
                )
            return

        if self.state_machine.current_state in (
            ActivationState.IDLE,
            ActivationState.FACE_DETECTED,
            ActivationState.TRACKING,
        ):
            self.state_machine.transition(
                baseline_state, now_ms, "baseline_update", force=True
            )

        stare_candidate = self._is_stare_candidate(tracking_data)
        stare_ready = self.transition_manager.update_stare(stare_candidate, delta_ms)

        if stare_ready and self.state_machine.current_state in (
            ActivationState.FACE_DETECTED,
            ActivationState.TRACKING,
        ):
            if self.state_machine.transition(
                ActivationState.STARE_DETECTED, now_ms, "stare_detected", force=True
            ):
                self.stare_timer.start(reset=True)
                self.event_dispatcher.emit(
                    "on_stare_start", self._build_payload(reason="stare_detected")
                )
        elif (
            not stare_candidate
            and self.state_machine.current_state == ActivationState.STARE_DETECTED
        ):
            self.stare_timer.reset()
            self.state_machine.transition(
                baseline_state, now_ms, "stare_lost", force=True
            )

    def _process_activation_progress(self, now_ms: float) -> None:
        if self.state_machine.current_state == ActivationState.STARE_DETECTED:
            if self.stare_timer.elapsed_ms >= self.config.stare_duration_ms:
                if self.state_machine.transition(
                    ActivationState.ACTIVATING, now_ms, "activation_started", force=True
                ):
                    self.activation_timer.start(reset=True)
                    self.event_dispatcher.emit(
                        "on_activation_start",
                        self._build_payload(reason="activation_started"),
                    )

        if self.state_machine.current_state == ActivationState.ACTIVATING:
            effective_duration_ms = max(
                1.0,
                self.config.activation_duration_ms
                / max(0.01, self.config.progression_speed),
            )
            self.snapshot.activation_elapsed_ms = self.activation_timer.elapsed_ms
            self.snapshot.activation_progress = min(
                100.0,
                (self.activation_timer.elapsed_ms / effective_duration_ms) * 100.0,
            )
            self.snapshot.activation_ready = self.snapshot.activation_progress >= 100.0

            if self.snapshot.activation_ready:
                self.activation_count += 1
                self.snapshot.activation_count = self.activation_count
                if self.state_machine.transition(
                    ActivationState.POWER_ACTIVE,
                    now_ms,
                    "activation_complete",
                    force=True,
                ):
                    self.event_dispatcher.emit(
                        "on_activation_complete",
                        self._build_payload(reason="activation_complete"),
                    )
                    self.cooldown_manager.start(self.config.cooldown_duration_ms)

        elif self.state_machine.current_state == ActivationState.POWER_ACTIVE:
            self.snapshot.activation_progress = 100.0
            self.snapshot.activation_ready = True
            self.snapshot.activation_elapsed_ms = max(
                self.snapshot.activation_elapsed_ms, self.activation_timer.elapsed_ms
            )
            if not self.cooldown_manager.timer.running:
                self.cooldown_manager.start(self.config.cooldown_duration_ms)

        elif self.state_machine.current_state == ActivationState.COOLDOWN:
            self.snapshot.activation_ready = False
            self.snapshot.activation_progress = 0.0
            self.snapshot.cooldown_remaining_ms = self.cooldown_manager.remaining_ms

        elif self.state_machine.current_state in (
            ActivationState.IDLE,
            ActivationState.FACE_DETECTED,
            ActivationState.TRACKING,
        ):
            self.snapshot.activation_progress = 0.0
            self.snapshot.activation_ready = False
            self.snapshot.activation_elapsed_ms = 0.0

    def _enter_interrupted(self, now_ms: float, reason: str) -> None:
        if self.state_machine.transition(
            ActivationState.INTERRUPTED, now_ms, reason, force=True
        ):
            self.event_dispatcher.emit(
                "on_interrupt", self._build_payload(reason=reason)
            )
            self.stare_timer.reset()
            self.activation_timer.reset()
            self.cooldown_manager.start(self.config.cooldown_duration_ms)

    def _derive_baseline_state(
        self, tracking_data: Optional[TrackingFrameData]
    ) -> ActivationState:
        if tracking_data is None or not tracking_data.face.detected:
            return ActivationState.IDLE

        if (
            tracking_data.overall_confidence >= self.config.confidence_threshold
            and tracking_data.eye_contact.in_contact
        ):
            return ActivationState.TRACKING

        if tracking_data.face.detected:
            return ActivationState.FACE_DETECTED

        return ActivationState.IDLE

    def _is_stare_candidate(self, tracking_data: Optional[TrackingFrameData]) -> bool:
        if tracking_data is None:
            return False

        if not tracking_data.face.detected:
            return False

        if tracking_data.blink.is_blinking:
            return False

        if tracking_data.overall_confidence < self.config.confidence_threshold:
            return False

        return tracking_data.eye_contact.in_contact

    def _get_interrupt_reason(
        self, tracking_data: Optional[TrackingFrameData]
    ) -> Optional[str]:
        if tracking_data is None or not tracking_data.face.detected:
            return "face_lost"

        if tracking_data.blink.is_blinking:
            return "blink"

        if tracking_data.overall_confidence < self.config.confidence_threshold:
            return "confidence_drop"

        if not tracking_data.eye_contact.in_contact:
            return "gaze_break"

        return None

    def _refresh_snapshot(self, tracking_data: Optional[TrackingFrameData]) -> None:
        self.snapshot.state = self.state_machine.current_state
        self.snapshot.cooldown_remaining_ms = (
            self.cooldown_manager.remaining_ms
            if self.cooldown_manager.timer.running
            else 0.0
        )
        self.snapshot.stare_duration_ms = self.stare_timer.elapsed_ms
        self.snapshot.interruption_duration_ms = self.interruption_timer.elapsed_ms
        self.snapshot.transition_reason = self.state_machine.last_transition_reason
        self.snapshot.interruption_reason = (
            self.transition_manager.last_interrupt_reason
        )
        self.snapshot.activation_count = self.activation_count

        if tracking_data is not None:
            self.snapshot.details = {
                "face_confidence": tracking_data.face.confidence,
                "eye_contact_confidence": tracking_data.eye_contact.confidence,
                "gaze_centered_ratio": tracking_data.eye_contact.gaze_centered_ratio,
            }
        else:
            self.snapshot.details = {}

    def _build_payload(self, reason: str) -> Dict[str, Any]:
        return {
            "state": self.state_machine.current_state.value,
            "reason": reason,
            "snapshot": self.snapshot.as_dict(),
            "activation_count": self.activation_count,
        }
