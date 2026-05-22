"""
SHARINGAN CINEMA ENGINE - PHASE 1 through PHASE 8
Anime-inspired cinematic webcam system
Foundation: Webcam + Tracking + Activation + Effects + Presentation + Audio + Recording + Intelligence

Entry point for the application.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import sys
import traceback
from typing import Optional

from activation.activation_engine import ActivationEngine
from audio.audio_manager import AudioManager
from config.settings import AppConfig
from core.camera import CameraEngine
from core.fps_manager import FPSManager
from effects.effect_manager import EffectManager
from intelligence.cinematic_director import CinematicDirector
from presentation.presentation_manager import PresentationManager
from recording.recording_manager import RecordingManager
from tracking.debug_renderer import TrackingDebugRenderer
from tracking.tracking_engine import TrackingEngine
from ui.keyboard import KeyboardHandler
from ui.overlay import DebugOverlay
from ui.window import WindowManager
from utils.logger import Logger
from utils.processing import FrameProcessor


class SharinganEngine:
    """
    Main application engine orchestrating all subsystems.
    Coordinates camera, tracking, rendering, input, and performance management.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize Sharingan engine with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = Logger(name="SharinganEngine", verbose=config.verbose)

        # Initialize subsystems
        self.camera: Optional[CameraEngine] = None
        self.fps_manager: Optional[FPSManager] = None
        self.window: Optional[WindowManager] = None
        self.overlay: Optional[DebugOverlay] = None
        self.keyboard: Optional[KeyboardHandler] = None

        # Phase 2 tracking
        self.tracking_engine: Optional[TrackingEngine] = None
        self.tracking_renderer: Optional[TrackingDebugRenderer] = None
        self.activation_engine: Optional[ActivationEngine] = None
        self.effect_manager: Optional[EffectManager] = None
        self.presentation_manager: Optional[PresentationManager] = None
        self.audio_manager: Optional[AudioManager] = None

        # Phase 7 recording
        self.recording_manager: Optional[RecordingManager] = None

        # Phase 8 intelligence
        self.cinematic_director: Optional[CinematicDirector] = None

        # State
        self.is_running = False
        self.is_recording = False
        self.frame_count = 0
        self.show_tracking_debug = False
        self.show_debug_overlay = False

        self._initialize_subsystems()

    def _initialize_subsystems(self) -> None:
        """Initialize all engine subsystems."""
        self.logger.info("=" * 50)
        self.logger.info("SHARINGAN CINEMA ENGINE - PHASE 1 through 8")
        self.logger.info(
            "Webcam + Tracking + Activation + Effects + Presentation + Audio + Recording + Intelligence"
        )
        self.logger.info("=" * 50)

        try:
            # Camera
            self.logger.info("Initializing camera...")
            self.camera = CameraEngine(
                device_id=self.config.camera.device_id,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
                backend=self.config.camera.backend,
                logger=self.logger,
            )

            # FPS Manager
            self.logger.info("Initializing FPS manager...")
            self.fps_manager = FPSManager(
                smoothing_window=self.config.performance.fps_smoothing_window,
                fps_cap=self.config.performance.fps_cap,
                logger=self.logger,
            )

            # Window
            self.logger.info("Initializing window...")
            self.window = WindowManager(
                window_name=self.config.window.title,
                width=self.config.camera.width,
                height=self.config.camera.height,
                resizable=self.config.window.resizable,
                logger=self.logger,
            )

            # Overlay
            self.logger.info("Initializing debug overlay...")
            self.overlay = DebugOverlay(
                enabled=self.config.performance.enable_debug, logger=self.logger
            )

            # Keyboard
            self.logger.info("Initializing keyboard handler...")
            self.keyboard = KeyboardHandler(logger=self.logger)
            self._setup_keyboard_bindings()

            # Phase 2: Tracking
            if self.config.tracking.enabled:
                self.logger.info("Initializing tracking engine (Phase 2)...")
                self.tracking_engine = TrackingEngine(
                    smoothing_factor=self.config.tracking.landmark_smoothing_factor,
                    logger=self.logger,
                )

                # Apply tracking configuration
                self.tracking_engine.update_config(
                    blink_threshold=self.config.tracking.blink_threshold,
                    eye_contact_duration_ms=self.config.tracking.eye_contact_duration_threshold_ms,
                    smoothing_factor=self.config.tracking.landmark_smoothing_factor,
                )

                # Initialize tracking debug renderer
                self.tracking_renderer = TrackingDebugRenderer(
                    enabled=self.config.tracking.enable_tracking_debug,
                    logger=self.logger,
                )

            if self.config.activation.enabled:
                self.logger.info("Initializing activation engine (Phase 3)...")
                self.activation_engine = ActivationEngine(
                    config=self.config.activation, logger=self.logger
                )
                self._setup_activation_bindings()

            if self.config.effects.enabled and self.activation_engine:
                self.logger.info("Initializing effects engine (Phase 4)...")
                self.effect_manager = EffectManager(
                    config=self.config.effects, logger=self.logger
                )
                self.effect_manager.bind_activation_engine(self.activation_engine)

            if self.config.presentation.enabled and self.activation_engine:
                self.logger.info("Initializing presentation engine (Phase 5)...")
                self.presentation_manager = PresentationManager(
                    config=self.config.presentation, logger=self.logger
                )
                self.presentation_manager.bind_activation_engine(self.activation_engine)

            if self.config.audio.enabled and self.activation_engine:
                self.logger.info("Initializing audio engine (Phase 6)...")
                self.audio_manager = AudioManager(
                    config=self.config.audio, logger=self.logger
                )
                self.audio_manager.bind_activation_engine(self.activation_engine)

            # Phase 7: Recording
            if self.config.recording.enabled:
                self.logger.info("Initializing recording engine (Phase 7)...")
                self.recording_manager = RecordingManager(
                    config=self.config.recording, logger=self.logger
                )
                if self.activation_engine:
                    self.recording_manager.bind_activation_engine(
                        self.activation_engine
                    )

            # Phase 8: Intelligence
            if self.config.intelligence.enabled:
                self.logger.info("Initializing cinematic intelligence (Phase 8)...")
                self.cinematic_director = CinematicDirector(
                    config=self.config.intelligence, logger=self.logger
                )
                if self.activation_engine:
                    self.cinematic_director.bind_activation_engine(
                        self.activation_engine
                    )

            self.logger.success("All subsystems initialized successfully")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            raise

    def _setup_keyboard_bindings(self) -> None:
        """Setup keyboard callbacks."""
        if self.keyboard is None:
            return

        self.keyboard.register_callback("quit", self.request_quit)
        self.keyboard.register_callback("toggle_fullscreen", self.toggle_fullscreen)
        self.keyboard.register_callback("toggle_recording", self.toggle_recording)
        self.keyboard.register_callback("toggle_debug", self.toggle_debug)
        self.keyboard.register_callback(
            "toggle_tracking_debug", self.toggle_tracking_debug
        )

        # Phase 7 recording controls
        self.keyboard.register_callback("take_screenshot", self.take_screenshot)
        self.keyboard.register_callback("capture_replay", self.capture_replay)
        self.keyboard.register_callback("toggle_creator_mode", self.toggle_creator_mode)
        self.keyboard.register_callback("quality_low", lambda: self.set_quality("LOW"))
        self.keyboard.register_callback(
            "quality_medium", lambda: self.set_quality("MEDIUM")
        )
        self.keyboard.register_callback(
            "quality_high", lambda: self.set_quality("HIGH")
        )
        self.keyboard.register_callback(
            "quality_cinematic", lambda: self.set_quality("CINEMATIC")
        )

        # Phase 8 intelligence controls
        self.keyboard.register_callback("cycle_personality", self.cycle_personality)
        self.keyboard.register_callback("force_rare_event", self.force_rare_event)
        self.keyboard.register_callback(
            "personality_calm", lambda: self.set_personality("CALM_OBSERVER")
        )
        self.keyboard.register_callback(
            "personality_corrupted", lambda: self.set_personality("CORRUPTED_ENTITY")
        )
        self.keyboard.register_callback(
            "personality_unstable", lambda: self.set_personality("UNSTABLE_POWER")
        )
        self.keyboard.register_callback(
            "personality_aggressive",
            lambda: self.set_personality("AGGRESSIVE_AWAKENING"),
        )
        self.keyboard.register_callback(
            "personality_void", lambda: self.set_personality("SILENT_VOID")
        )

    def _setup_activation_bindings(self) -> None:
        """Setup activation event callbacks."""
        if self.activation_engine is None:
            return

        self.activation_engine.on("on_stare_start", self._handle_stare_start)
        self.activation_engine.on("on_activation_start", self._handle_activation_start)
        self.activation_engine.on(
            "on_activation_complete", self._handle_activation_complete
        )
        self.activation_engine.on("on_interrupt", self._handle_activation_interrupt)
        self.activation_engine.on("on_cooldown_start", self._handle_cooldown_start)
        self.activation_engine.on("on_cooldown_end", self._handle_cooldown_end)

    def run(self) -> int:
        """
        Main engine loop. Runs until quit is requested.

        Returns:
            Exit code (0 = success, 1 = error)
        """
        if (
            not self.camera
            or not self.window
            or not self.keyboard
            or not self.fps_manager
            or not self.overlay
        ):
            self.logger.error("Engine not properly initialized")
            return 1

        self.is_running = True
        self.logger.info("Starting main loop...")
        self.logger.info(
            "Press 'Q' quit | 'F' fullscreen | 'D' debug | 'T' tracking | 'R' recording"
        )
        self.logger.info(
            "Press 'P' screenshot | 'L' replay | 'C' creator | '1-4' quality"
        )
        self.logger.info(
            "Press 'N' cycle personality | 'M' rare event | '5-9' personalities"
        )

        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    self.logger.error("Failed to capture frame")
                    break

                self.frame_count += 1

                # Update FPS
                fps = self.fps_manager.tick()

                # Process frame (future: add effects here)
                # For now, just apply mirror flip for natural appearance
                base_frame = FrameProcessor.flip_horizontal(frame)
                base_frame = FrameProcessor.enhance_visibility(base_frame)
                processed_frame = base_frame

                # Phase 2: Face/Iris tracking (detection - separate from rendering)
                tracking_data = None
                if self.tracking_engine and self.config.tracking.enabled:
                    tracking_data = self.tracking_engine.process_frame(processed_frame)

                activation_data = None
                if self.activation_engine and self.config.activation.enabled:
                    activation_data = self.activation_engine.update(tracking_data)

                    # Removed instant override to allow cinematic buildup
                effect_context = None
                if self.effect_manager and self.config.effects.enabled:
                    camera_info = self.camera.get_frame_info()
                    resolution = (camera_info["width"], camera_info["height"])
                    effect_context = self.effect_manager.update(
                        tracking_data, activation_data, resolution, processed_frame
                    )
                    processed_frame = self.effect_manager.render(
                        processed_frame, effect_context
                    )

                presentation_context = None
                if self.presentation_manager and self.config.presentation.enabled:
                    camera_info = self.camera.get_frame_info()
                    resolution = (camera_info["width"], camera_info["height"])
                    presentation_context = self.presentation_manager.update(
                        tracking_data, activation_data, resolution
                    )
                    processed_frame = self.presentation_manager.render(
                        processed_frame, presentation_context
                    )

                processed_frame = FrameProcessor.ensure_visible_frame(
                    processed_frame, base_frame
                )

                audio_data = None
                if self.audio_manager and self.config.audio.enabled:
                    audio_data = self.audio_manager.update(activation_data)

                # ── Phase 8: Intelligence update ──
                director_snapshot = None
                if self.cinematic_director and self.config.intelligence.enabled:
                    director_snapshot = self.cinematic_director.update(
                        tracking_data, activation_data
                    )

                # ── Phase 7: Feed final composited frame to recording pipeline ──
                # This happens AFTER all effects/presentation but BEFORE debug overlay
                # so that recordings capture the cinematic output, not debug text.
                if self.recording_manager:
                    self.recording_manager.push_frame(processed_frame)

                # ── Creator mode: suppress debug overlays for clean footage ──
                creator_mode_active = (
                    self.recording_manager is not None
                    and self.recording_manager.is_creator_mode
                )

                # Render debug overlay (suppressed in creator mode)
                camera_info = self.camera.get_frame_info()
                resolution = (camera_info["width"], camera_info["height"])

                active_states = {
                    "fullscreen": self.window.is_fullscreen,
                    "recording": self.is_recording,
                }

                # Update active states from tracking
                if tracking_data:
                    active_states["tracking"] = tracking_data.state.value != "NO_FACE"
                    active_states["eye_contact"] = tracking_data.eye_contact.in_contact
                    active_states["blinking"] = tracking_data.blink.is_blinking

                if activation_data:
                    active_states["activation"] = activation_data.state.value not in (
                        "IDLE",
                        "FACE_DETECTED",
                        "TRACKING",
                        "COOLDOWN",
                    )
                    active_states["power_active"] = activation_data.activation_ready
                    active_states["cooldown"] = (
                        activation_data.state.value == "COOLDOWN"
                    )
                if self.audio_manager:
                    active_states["audio"] = self.audio_manager.is_active()

                if self.recording_manager:
                    active_states["recording"] = self.recording_manager.is_recording
                    if creator_mode_active:
                        active_states["creator"] = True

                debug_lines = []
                if not creator_mode_active:
                    if self.activation_engine and activation_data:
                        debug_lines.extend(self.activation_engine.get_debug_lines())
                    if self.effect_manager and effect_context:
                        if debug_lines:
                            debug_lines.append("")
                        debug_lines.extend(self.effect_manager.get_debug_lines())
                    if self.presentation_manager and presentation_context:
                        if debug_lines:
                            debug_lines.append("")
                        debug_lines.extend(self.presentation_manager.get_debug_lines())
                    if self.audio_manager and audio_data:
                        if debug_lines:
                            debug_lines.append("")
                        debug_lines.extend(self.audio_manager.get_debug_lines())
                    if self.recording_manager:
                        if debug_lines:
                            debug_lines.append("")
                        debug_lines.extend(self.recording_manager.get_debug_lines())
                    if self.cinematic_director and director_snapshot:
                        if debug_lines:
                            debug_lines.append("")
                        debug_lines.extend(self.cinematic_director.get_debug_lines())

                if creator_mode_active:
                    # In creator mode, only show a minimal recording indicator
                    display_frame = processed_frame.copy()
                    if self.recording_manager.is_recording:
                        self._render_recording_indicator(display_frame)
                else:
                    display_frame = self.overlay.render(
                        processed_frame,
                        fps=fps,
                        resolution=resolution,
                        recording=self.is_recording,
                        active_states=active_states,
                        custom_lines=debug_lines or None,
                    )

                # Phase 2: Render tracking debug (rendering - separate from detection)
                if not creator_mode_active:
                    if (
                        self.tracking_renderer
                        and tracking_data
                        and self.show_tracking_debug
                    ):
                        display_frame = self.tracking_renderer.render(
                            display_frame, tracking_data
                        )

                # Display frame
                self.window.display_frame(display_frame)

                # Handle input
                self.keyboard.handle_input(wait_ms=1)

                # Frame rate limiting
                self.fps_manager.wait_for_frame_interval()

                # Check if window is still open
                if not self.window.is_window_open():
                    self.logger.info("Window closed by user")
                    break

            return 0

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Engine error: {e}")
            traceback.print_exc()
            return 1
        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    # Recording indicator for creator mode
    # ------------------------------------------------------------------

    @staticmethod
    def _render_recording_indicator(frame) -> None:
        """Draw a small red recording dot in the top-right corner."""
        import cv2

        h, w = frame.shape[:2]
        center = (w - 30, 30)
        cv2.circle(frame, center, 10, (0, 0, 255), -1)
        cv2.circle(frame, center, 12, (0, 0, 200), 2)

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def request_quit(self) -> None:
        """Request engine shutdown."""
        self.logger.info("Quit requested")
        self.is_running = False

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if self.window:
            self.window.toggle_fullscreen()

    def toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.recording_manager:
            self.recording_manager.toggle_recording()
            self.is_recording = self.recording_manager.is_recording
        else:
            self.is_recording = not self.is_recording
            state = "started" if self.is_recording else "stopped"
            self.logger.info(f"Recording {state}")

    def toggle_debug(self) -> None:
        """Toggle debug overlay."""
        if self.overlay:
            self.overlay.toggle()
            self.show_debug = self.overlay.enabled

    def toggle_tracking_debug(self) -> None:
        """Toggle tracking debug visualization."""
        if self.tracking_renderer:
            self.tracking_renderer.toggle()
            self.show_tracking_debug = self.tracking_renderer.enabled

    def take_screenshot(self) -> None:
        """Capture a screenshot of the current composited frame."""
        if self.recording_manager:
            self.recording_manager.take_screenshot()
        else:
            self.logger.warning("Recording manager not available")

    def capture_replay(self) -> None:
        """Capture and export a replay clip."""
        if self.recording_manager:
            self.recording_manager.export_replay()
        else:
            self.logger.warning("Recording manager not available")

    def toggle_creator_mode(self) -> None:
        """Toggle creator mode (clean output for content creation)."""
        if self.recording_manager:
            self.recording_manager.toggle_creator_mode()
        else:
            self.logger.warning("Recording manager not available")

    def cycle_personality(self) -> None:
        """Cycle to the next personality mode."""
        if self.cinematic_director:
            self.cinematic_director.cycle_personality()
        else:
            self.logger.warning("Cinematic director not available")

    def set_personality(self, mode: str) -> None:
        """Set a specific personality mode."""
        if self.cinematic_director:
            self.cinematic_director.set_personality(mode)
            self.logger.info(f"Personality set to {mode}")
        else:
            self.logger.warning("Cinematic director not available")

    def force_rare_event(self) -> None:
        """Force trigger a rare cinematic event."""
        if self.cinematic_director:
            self.cinematic_director.force_rare_event()
        else:
            self.logger.warning("Cinematic director not available")

    def set_quality(self, mode: str) -> None:
        """Set quality mode (LOW/MEDIUM/HIGH/CINEMATIC)."""
        if self.recording_manager:
            self.recording_manager.set_quality_mode(mode)
            self.logger.info(f"Quality mode set to {mode}")
        else:
            self.logger.warning("Recording manager not available")

    # ------------------------------------------------------------------
    # Activation event handlers
    # ------------------------------------------------------------------

    def _handle_stare_start(self, payload: dict) -> None:
        self.logger.info("Activation stare detected")

    def _handle_activation_start(self, payload: dict) -> None:
        self.logger.info("Activation buildup started")

    def _handle_activation_complete(self, payload: dict) -> None:
        self.logger.success("Activation completed")

    def _handle_activation_interrupt(self, payload: dict) -> None:
        reason = payload.get("reason", "unknown")
        self.logger.info(f"Activation interrupted ({reason})")

    def _handle_cooldown_start(self, payload: dict) -> None:
        self.logger.debug("Activation cooldown started")

    def _handle_cooldown_end(self, payload: dict) -> None:
        self.logger.debug("Activation cooldown ended")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        self.logger.info("Shutting down...")

        # Stop recording first to finalize files
        if self.recording_manager:
            self.recording_manager.release()

        if self.cinematic_director:
            self.cinematic_director.release()

        if self.camera:
            self.camera.release()

        if self.tracking_engine:
            self.tracking_engine.release()

        if self.effect_manager:
            self.effect_manager.release()

        if self.presentation_manager:
            self.presentation_manager.release()

        if self.audio_manager:
            self.audio_manager.release()

        if self.window:
            self.window.close()

        self.logger.success(f"Shutdown complete. Processed {self.frame_count} frames")


def main() -> int:
    """
    Application entry point.

    Returns:
        Exit code
    """
    try:
        # Load configuration
        config = AppConfig.default()

        # Create and run engine
        engine = SharinganEngine(config)
        exit_code = engine.run()

        return exit_code

    except Exception as e:
        logger = Logger(name="Main")
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
