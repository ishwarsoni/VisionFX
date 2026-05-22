"""Lightweight event dispatcher for activation hooks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List

ActivationEventHandler = Callable[[Dict[str, Any]], None]


class ActivationEventDispatcher:
    """Simple publish/subscribe dispatcher."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[ActivationEventHandler]] = defaultdict(
            list
        )

    def on(self, event_name: str, handler: ActivationEventHandler) -> None:
        if handler not in self._handlers[event_name]:
            self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: ActivationEventHandler) -> None:
        handlers = self._handlers.get(event_name)
        if not handlers:
            return
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        for handler in list(self._handlers.get(event_name, [])):
            try:
                handler(payload)
            except Exception:
                continue
