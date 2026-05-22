"""
Keyboard input handling system.
"""

from typing import Callable, Dict, Optional

import cv2

from utils.logger import Logger


class KeyboardHandler:
    """
    Production-grade keyboard input handler.
    Manages key bindings and provides clean control interface.
    """

    # Default key bindings
    DEFAULT_BINDINGS = {
        ord("q"): "quit",
        ord("Q"): "quit",
        ord("f"): "toggle_fullscreen",
        ord("F"): "toggle_fullscreen",
        ord("r"): "toggle_recording",
        ord("R"): "toggle_recording",
        ord("d"): "toggle_debug",
        ord("D"): "toggle_debug",
        ord("t"): "toggle_tracking_debug",
        ord("T"): "toggle_tracking_debug",
        ord("p"): "take_screenshot",
        ord("P"): "take_screenshot",
        ord("l"): "capture_replay",
        ord("L"): "capture_replay",
        ord("c"): "toggle_creator_mode",
        ord("C"): "toggle_creator_mode",
        ord("1"): "quality_low",
        ord("2"): "quality_medium",
        ord("3"): "quality_high",
        ord("4"): "quality_cinematic",
        ord("n"): "cycle_personality",
        ord("N"): "cycle_personality",
        ord("m"): "force_rare_event",
        ord("M"): "force_rare_event",
        ord("5"): "personality_calm",
        ord("6"): "personality_corrupted",
        ord("7"): "personality_unstable",
        ord("8"): "personality_aggressive",
        ord("9"): "personality_void",
    }

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize keyboard handler.

        Args:
            logger: Logger instance
        """
        self.logger = logger or Logger(name="KeyboardHandler")
        self.bindings: Dict[int, str] = self.DEFAULT_BINDINGS.copy()
        self.callbacks: Dict[str, Callable] = {}
        self.last_key = None

        self.logger.info("Keyboard handler initialized")
        self._log_bindings()

    def _log_bindings(self) -> None:
        """Log key bindings for reference."""
        binding_strs = []
        for key, action in self.DEFAULT_BINDINGS.items():
            key_char = chr(key) if key < 128 else f"code_{key}"
            binding_strs.append(f"{key_char}→{action}")
        self.logger.debug(f"Key bindings: {', '.join(binding_strs)}")

    def register_callback(self, action: str, callback: Callable) -> None:
        """
        Register callback for an action.

        Args:
            action: Action name (e.g., 'quit', 'toggle_fullscreen')
            callback: Callable to execute
        """
        self.callbacks[action] = callback
        self.logger.debug(f"Registered callback for action: {action}")

    def process_input(self, wait_ms: int = 1) -> Optional[str]:
        """
        Process keyboard input and return triggered action.

        Args:
            wait_ms: Milliseconds to wait for key press

        Returns:
            Action string if key was pressed, None otherwise
        """
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:  # No key pressed
            return None

        self.last_key = key

        if key in self.bindings:
            action = self.bindings[key]
            return action

        return None

    def handle_input(self, wait_ms: int = 1) -> Optional[str]:
        """
        Process input and trigger callbacks.

        Args:
            wait_ms: Milliseconds to wait for key press

        Returns:
            Action that was triggered, or None
        """
        action = self.process_input(wait_ms)

        if action is None:
            return None

        if action in self.callbacks:
            try:
                self.callbacks[action]()
            except Exception as e:
                self.logger.error(f"Callback error for action '{action}': {e}")

        return action

    def rebind_key(self, key_code: int, action: str) -> None:
        """
        Rebind a key to a different action.

        Args:
            key_code: ASCII key code
            action: Action name
        """
        old_action = self.bindings.get(key_code, "none")
        self.bindings[key_code] = action
        self.logger.debug(f"Rebound key {chr(key_code)}: {old_action} → {action}")

    def get_last_key(self) -> Optional[int]:
        """Get the last key code that was pressed."""
        return self.last_key
