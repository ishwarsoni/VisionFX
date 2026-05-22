"""Lightweight session memory for cinematic continuity across activations."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from intelligence.intelligence_data import BehaviorMetrics, SessionMemory
from utils.logger import Logger


class MemorySystem:
    """
    Maintains a ``SessionMemory`` that accumulates across the current
    session and can optionally persist to a JSON file between runs.

    Memory drives personality adaptation and cinematic continuity — the
    system remembers how the user interacts and adjusts its cinematic
    identity accordingly.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        persist_enabled: bool = False,
        logger: Optional[Logger] = None,
    ) -> None:
        self.logger = logger or Logger(name="MemorySystem")
        self._persist_path = persist_path or "recordings/memory.json"
        self._persist_enabled = persist_enabled
        self._session_start = time.perf_counter()

        self.memory = SessionMemory()

        if self._persist_enabled:
            self._load()

        self.logger.info("MemorySystem initialized")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, behavior: BehaviorMetrics, activation_data=None) -> SessionMemory:
        """
        Integrate current behavior metrics into session memory.

        Args:
            behavior: Current frame's BehaviorMetrics.
            activation_data: Current ActivationSnapshot (or None).

        Returns:
            Updated ``SessionMemory``.
        """
        m = self.memory

        # Session time
        m.total_session_time_s = time.perf_counter() - self._session_start

        # Activations
        m.total_activations = behavior.activation_count

        # Longest stare
        if behavior.max_stare_duration_ms > m.longest_stare_ms:
            m.longest_stare_ms = behavior.max_stare_duration_ms

        # Interruption patterns
        if activation_data is not None and activation_data.interruption_reason:
            reason = activation_data.interruption_reason
            m.interruption_patterns[reason] = m.interruption_patterns.get(reason, 0) + 1

        return m

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def record_activation(self, duration_ms: float) -> None:
        """Record a completed activation duration."""
        self.memory.activation_history.append(duration_ms)
        # Keep history bounded
        if len(self.memory.activation_history) > 200:
            self.memory.activation_history = self.memory.activation_history[-200:]

    def record_rare_event(self) -> None:
        """Increment rare event counter."""
        self.memory.rare_events_triggered += 1

    def record_personality(self, personality_name: str) -> None:
        """Update dominant personality tracking."""
        self.memory.dominant_personality = personality_name

    def record_escalation_peak(self, peak: float) -> None:
        """Track peak escalation reached."""
        if peak > self.memory.peak_escalation:
            self.memory.peak_escalation = peak

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist memory to disk."""
        if not self._persist_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(self.memory.as_dict(), f, indent=2)
            self.logger.info(f"Memory saved to {self._persist_path}")
        except Exception as e:
            self.logger.warning(f"Memory save failed: {e}")

    def _load(self) -> None:
        """Load memory from disk if available."""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.memory.total_activations = data.get("total_activations", 0)
            self.memory.longest_stare_ms = data.get("longest_stare_ms", 0.0)
            self.memory.peak_escalation = data.get("peak_escalation", 0.0)
            self.memory.dominant_personality = data.get(
                "dominant_personality", "CALM_OBSERVER"
            )
            self.memory.rare_events_triggered = data.get("rare_events_triggered", 0)
            self.memory.interruption_patterns = data.get("interruption_patterns", {})
            self.logger.info(f"Memory loaded from {self._persist_path}")
        except Exception as e:
            self.logger.warning(f"Memory load failed: {e}")

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def get_debug_lines(self) -> list:
        m = self.memory
        return [
            f"MEM ACTS: {m.total_activations}",
            f"MEM STARE: {m.longest_stare_ms / 1000:.1f}s",
            f"MEM PEAK: {m.peak_escalation:.2f}",
            f"MEM RARE: {m.rare_events_triggered}",
            f"MEM TIME: {m.total_session_time_s:.0f}s",
        ]

    def release(self) -> None:
        """Save and release resources."""
        self.save()
