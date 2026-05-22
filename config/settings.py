"""
Centralized configuration system for the webcam engine.
"""

from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Camera configuration settings."""

    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "default"


@dataclass
class WindowConfig:
    """Window configuration settings."""

    title: str = "【 SHARINGAN CINEMA ENGINE 】"
    fullscreen: bool = False
    resizable: bool = True
    decorated: bool = True


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""

    fps_cap: int = 60
    frame_skip: int = 0
    enable_debug: bool = False
    fps_smoothing_window: int = 30


@dataclass
class TrackingConfig:
    """Face and iris tracking configuration (Phase 2)."""

    enabled: bool = True
    landmark_smoothing_factor: float = 0.7
    confidence_smoothing_factor: float = 0.8
    blink_threshold: float = 0.3
    min_blink_duration_ms: float = 50.0
    max_blink_duration_ms: float = 400.0
    gaze_center_threshold: float = 0.6
    iris_symmetry_threshold: float = 0.7
    eye_contact_duration_threshold_ms: float = 200.0
    min_tracking_confidence: float = 0.5
    enable_tracking_debug: bool = False


@dataclass
class ActivationConfig:
    """Activation intelligence configuration (Phase 3)."""

    enabled: bool = True
    stare_duration_ms: float = 1800.0
    activation_duration_ms: float = 2500.0
    cooldown_duration_ms: float = 1500.0
    interruption_threshold_ms: float = 180.0
    confidence_threshold: float = 0.65
    progression_speed: float = 1.0
    transition_stability_ms: float = 80.0


@dataclass
class EffectsConfig:
    """Cinematic visual effects configuration (Phase 4)."""

    enabled: bool = True
    enable_debug: bool = False
    compositor_quality: int = 2
    enable_activation_impact: bool = False
    enable_eye_overlay: bool = True
    enable_genjutsu_glitch: bool = False
    enable_color_grading: bool = False
    enable_world_distortion: bool = False
    enable_screen_fx: bool = False
    glitch_intensity: float = 0.75
    bloom_strength: float = 0.18
    distortion_amount: float = 0.08
    eye_glow_intensity: float = 0.85
    screen_fx_intensity: float = 0.65
    impact_strength: float = 0.9


@dataclass
class PresentationConfig:
    """Cinematic presentation and immersion configuration (Phase 5)."""

    enabled: bool = True
    enable_debug: bool = True
    hud_intensity: float = 0.85
    cinematic_zoom_strength: float = 0.12
    corruption_intensity: float = 0.7
    immersion_strength: float = 0.55
    text_animation_ms: float = 520.0
    transition_fade_ms: float = 220.0
    transition_pulse_ms: float = 180.0
    transition_glitch_ms: float = 160.0
    transition_zoom_ms: float = 240.0
    stage_hold_ms: float = 120.0


@dataclass
class AudioConfig:
    """Real-time audio and synchronization configuration (Phase 6)."""

    enabled: bool = True
    enable_debug: bool = True
    backend: str = "auto"
    asset_root: str = "assets/audio"
    master_volume: float = 0.85
    ambience_volume: float = 0.22
    activation_volume: float = 0.9
    glitch_volume: float = 0.7
    ui_volume: float = 0.65
    transition_fade_ms: float = 180.0
    sync_lead_ms: float = 24.0
    activation_intensity: float = 0.88
    glitch_intensity: float = 0.72
    spatialization_intensity: float = 0.16


@dataclass
class RecordingConfig:
    """Recording, export, and content creation configuration (Phase 7)."""

    enabled: bool = True
    enable_debug: bool = True
    recording_fps: int = 30
    recording_codec: str = "mp4v"
    recording_format: str = "mp4"
    recording_resolution_scale: float = 1.0
    replay_duration_s: float = 5.0
    replay_buffer_enabled: bool = True
    screenshot_quality: int = 95
    screenshot_format: str = "png"
    max_queue_depth: int = 90
    quality_mode: str = "HIGH"
    creator_mode: bool = False
    export_root: str = "recordings"


@dataclass
class IntelligenceConfig:
    """Adaptive cinematic intelligence configuration (Phase 8)."""

    enabled: bool = True
    enable_debug: bool = True
    initial_personality: str = "CALM_OBSERVER"
    auto_adapt: bool = True
    escalation_rise_rate: float = 0.08
    escalation_decay_rate: float = 0.03
    rare_event_probability: float = 1.0
    rare_event_cooldown_s: float = 20.0
    immersion_intensity: float = 1.0
    memory_persistence: bool = False
    behavior_window_s: float = 30.0
    baseline_tension: float = 0.1


@dataclass
class AppConfig:
    """Main application configuration."""

    camera: CameraConfig
    window: WindowConfig
    performance: PerformanceConfig
    tracking: TrackingConfig
    activation: ActivationConfig
    effects: EffectsConfig
    presentation: PresentationConfig
    audio: AudioConfig
    recording: RecordingConfig
    intelligence: IntelligenceConfig
    headless: bool = False
    verbose: bool = True

    @staticmethod
    def default() -> "AppConfig":
        """Create default application configuration."""
        return AppConfig(
            camera=CameraConfig(),
            window=WindowConfig(),
            performance=PerformanceConfig(),
            tracking=TrackingConfig(),
            activation=ActivationConfig(),
            effects=EffectsConfig(),
            presentation=PresentationConfig(),
            audio=AudioConfig(),
            recording=RecordingConfig(),
            intelligence=IntelligenceConfig(),
        )
