"""Ordered compositor for stacking cinematic effects."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from effects.base_effect import BaseEffect
from effects.effect_data import EffectContext, EffectDebugState


class EffectCompositor:
    """Applies enabled effects in priority order."""

    def __init__(self):
        self.effects: List[BaseEffect] = []

    def register(self, effect: BaseEffect) -> None:
        self.effects.append(effect)
        self.effects.sort(key=lambda item: item.priority)

    def iter_enabled(self, context: EffectContext) -> Iterable[BaseEffect]:
        for effect in self.effects:
            if not effect.enabled:
                continue
            if context.quality < effect.minimum_quality:
                continue
            yield effect

    def render(
        self, frame: np.ndarray, context: EffectContext, debug_state: EffectDebugState
    ) -> np.ndarray:
        result = frame
        debug_state.compositor_order = []
        debug_state.active_effects = []
        debug_state.effect_intensities = {}

        for effect in self.iter_enabled(context):
            debug_state.compositor_order.append(effect.name)
            effect.update(context)
            if effect.is_active(context):
                debug_state.active_effects.append(effect.name)
            result = effect.process(result, context)
            intensity = getattr(effect, "current_intensity", None)
            if isinstance(intensity, (int, float)):
                debug_state.effect_intensities[effect.name] = float(intensity)

        return result
