"""Top-level orchestrator for anime power effects."""

from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np

from activation.activation_data import ActivationSnapshot
from activation.activation_engine import ActivationEngine
from config.settings import EffectsConfig
from effects.activation_impact import ActivationImpactEffect
from effects.anime_power_effect import AnimePowerEffect
from effects.base_effect import BaseEffect
from effects.color_grading import ColorGradingEffect
from effects.compositor import EffectCompositor
from effects.effect_data import EffectContext, EffectDebugState
from effects.hand_power_detector import HandPowerDetector
from effects.screen_fx import ScreenFxEffect
from effects.world_distortion import WorldDistortionEffect
from tracking.tracking_data import TrackingFrameData
from utils.logger import Logger


class EffectManager:
    """Bridges activation events to the composited visual effects pipeline."""

    def __init__(self, config: EffectsConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger(name="EffectManager")

        self.compositor = EffectCompositor()
        self.debug_state = EffectDebugState()
        self.frame_index = 0
        self.last_update_time_ms: Optional[float] = None
        self.render_time_ms = 0.0
        self.activation_engine: Optional[ActivationEngine] = None

        self.effects: List[BaseEffect] = []
        self._initialize_effects()

        self.logger.info("EffectManager initialized successfully")

    def _initialize_effects(self) -> None:
        effect_instances: List[BaseEffect] = []

        if self.config.enable_activation_impact:
            effect_instances.append(
                ActivationImpactEffect(
                    enabled=True, strength=self.config.impact_strength
                )
            )
        if self.config.enable_color_grading:
            effect_instances.append(
                ColorGradingEffect(
                    enabled=True, bloom_strength=self.config.bloom_strength
                )
            )
        if self.config.enable_world_distortion:
            effect_instances.append(
                WorldDistortionEffect(
                    enabled=True, distortion_amount=self.config.distortion_amount
                )
            )
        if self.config.enable_screen_fx:
            effect_instances.append(
                ScreenFxEffect(enabled=True, intensity=self.config.screen_fx_intensity)
            )

        self.anime_power_effect = AnimePowerEffect(enabled=True)
        self.hand_detector = HandPowerDetector()
        self.anime_power_effect.set_hand_detector(self.hand_detector)
        effect_instances.append(self.anime_power_effect)

        for effect in effect_instances:
            self.register_effect(effect)

    def register_effect(self, effect: BaseEffect) -> None:
        self.effects.append(effect)
        self.compositor.register(effect)

    def bind_activation_engine(self, activation_engine: ActivationEngine) -> None:
        self.activation_engine = activation_engine
        for event_name in (
            "on_stare_start",
            "on_activation_start",
            "on_activation_complete",
            "on_interrupt",
            "on_cooldown_start",
            "on_cooldown_end",
        ):
            activation_engine.on(
                event_name,
                lambda payload, event_name=event_name: self.handle_event(
                    event_name, payload
                ),
            )

    def handle_event(self, event_name: str, payload: dict) -> None:
        for effect in self.effects:
            effect.handle_event(event_name, payload)

    def update(
        self,
        tracking_data: Optional[TrackingFrameData],
        activation_data: Optional[ActivationSnapshot],
        frame_size: tuple,
        raw_frame: Any = None,
    ) -> EffectContext:
        now_ms = time.perf_counter() * 1000.0
        delta_ms = (
            0.0
            if self.last_update_time_ms is None
            else max(0.0, now_ms - self.last_update_time_ms)
        )
        self.last_update_time_ms = now_ms
        self.frame_index += 1

        context = EffectContext(
            frame_index=self.frame_index,
            timestamp_ms=now_ms,
            delta_ms=delta_ms,
            frame_size=frame_size,
            quality=max(1, min(3, int(self.config.compositor_quality))),
            debug_enabled=self.config.enable_debug,
            tracking_data=tracking_data,
            activation_data=activation_data,
            activation_state=activation_data.state if activation_data else None,
            activation_progress=(
                activation_data.activation_progress if activation_data else 0.0
            ),
            activation_ready=(
                activation_data.activation_ready if activation_data else False
            ),
            raw_frame=raw_frame,
        )

        return context

    def render(self, frame: np.ndarray, context: EffectContext) -> np.ndarray:
        if not self.config.enabled:
            self.debug_state = EffectDebugState(frame_index=context.frame_index)
            return frame

        if hasattr(self, "quality_enhancer") and context.activation_progress > 10:
            frame = self.quality_enhancer.enhance(frame)

        start_ms = time.perf_counter() * 1000.0
        result = self.compositor.render(frame, context, self.debug_state)
        self.render_time_ms = time.perf_counter() * 1000.0 - start_ms
        self.debug_state.render_time_ms = self.render_time_ms
        self.debug_state.frame_index = context.frame_index
        return result

    def get_debug_lines(self) -> List[str]:
        if not self.config.enable_debug:
            return []
        lines = [f"FX TIME: {self.render_time_ms:.2f}ms"]
        lines.extend(self.debug_state.lines())
        return lines

    def release(self) -> None:
        for effect in self.effects:
            if hasattr(effect, "release"):
                effect.release()
        self.effects.clear()
        self.compositor.effects.clear()
