<p align="center">
  <h1 align="center">SHARINGAN CINEMA ENGINE</h1>
  <p align="center">
    Real-time anime-inspired VFX engine with hand gesture tracking, GPU-accelerated jutsu effects, and cinematic intelligence.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/opencv-4.10-green?style=flat-square" alt="OpenCV">
  <img src="https://img.shields.io/badge/mediapipe-0.10-orange?style=flat-square" alt="MediaPipe">
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" alt="License">
</p>

---

## Overview

Sharingan Cinema Engine is a real-time computer vision application that transforms a standard webcam feed into an anime-inspired cinematic experience. Using MediaPipe for face and hand tracking, the engine renders Naruto-style jutsu effects — **Rasengan** and **Chidori** — directly onto the user's hands in real time, with optional GPU-accelerated shaders via ModernGL.

The system tracks the user's face and eyes for Sharingan overlay activation, detects hand gestures for power effects, and wraps the entire experience in a cinematic presentation layer with HUD elements, audio synchronization, recording capabilities, and adaptive AI-driven pacing.

---

## Features

| Category | Capabilities |
|---|---|
| **Jutsu Effects** | Rasengan (blue energy orb), Chidori (dense lightning blade), dual-hand fusion mode |
| **Hand Tracking** | MediaPipe Hand Landmarker with 21-point skeleton, handedness detection, dual-hand support |
| **Face & Eye Tracking** | MediaPipe FaceMesh, iris tracking, blink detection, eye contact detection |
| **Sharingan Overlay** | Eyelid-aware eye texture compositing with Mangekyo variants |
| **GPU Rendering** | ModernGL fragment shader for procedural lightning (plasma, FBM noise, bloom) |
| **Cinematic Presentation** | HUD overlays, cinematic camera zoom, corruption effects, animated text cues |
| **Audio System** | Activation-synced sound effects, ambient audio, glitch audio, spatial mixing |
| **Recording** | Video recording, instant replay buffer, screenshots, quality presets, creator mode |
| **AI Intelligence** | Personality engine, behavior tracking, escalation system, rare events, cinematic pacing |
| **Performance** | FPS management, frame rate capping, configurable resolution, low-latency camera pipeline |

---

## System Architecture

The engine follows a **phased subsystem architecture** where the main `SharinganEngine` orchestrates 8 specialized modules in a frame-by-frame pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SharinganEngine                              │
│                       (main.py - orchestrator)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │  Camera   │→│   Tracking   │→│  Activation   │                 │
│   │  Engine   │  │   Engine     │  │   Engine      │                 │
│   │ (Phase 1) │  │  (Phase 2)   │  │  (Phase 3)    │                 │
│   └──────────┘  └──────────────┘  └──────────────┘                 │
│        │                                   │                        │
│        ▼                                   ▼                        │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │  Effect   │→│ Presentation │→│    Audio      │                 │
│   │  Manager  │  │   Manager    │  │   Manager     │                 │
│   │ (Phase 4) │  │  (Phase 5)   │  │  (Phase 6)    │                 │
│   └──────────┘  └──────────────┘  └──────────────┘                 │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐  ┌──────────────┐                                   │
│   │ Recording │→│  Cinematic   │                                   │
│   │  Manager  │  │  Director    │                                   │
│   │ (Phase 7) │  │  (Phase 8)   │                                   │
│   └──────────┘  └──────────────┘                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Per-Frame Pipeline

```
Webcam → Capture → Mirror Flip → Face/Iris Tracking → Activation State Machine
  → Hand Detection → Gesture Classification → Jutsu Effect Rendering
  → Presentation Layer → Audio Sync → Recording Buffer → Display
```

### Tracking Pipeline (Phase 2)

The `TrackingEngine` coordinates four subsystems each frame:

1. **FaceTracker** — MediaPipe FaceMesh 468-landmark detection with confidence smoothing.
2. **IrisTracker** — Extracts iris center positions and eye openness from face mesh eye landmarks.
3. **BlinkDetector** — Detects blinks from eye aspect ratio with configurable threshold and duration windows.
4. **EyeContactDetector** — Determines if the user is looking at the camera using gaze centering and iris symmetry analysis.

All landmarks pass through an exponential moving average smoother to eliminate jitter.

### Effect Pipeline (Phase 4)

The `EffectManager` uses a compositor pattern. Each effect implements `BaseEffect.process()` and is rendered in priority order:

- **AnimePowerEffect** — Top-level hand power orchestrator. Runs the `HandPowerDetector` (MediaPipe Hands) each frame, assigns powers by handedness, and delegates rendering to `RasenganEffect` and `ChidoriEffect`.
- **RasenganEffect** — Layered blue-white energy orb with spiral wisps, inner threads, and environmental light spill. Pure OpenCV rendering.
- **ChidoriEffect** — Dense procedural lightning with recursive branching arcs, full-hand electricity coverage, particle system, environmental illumination, and screen shake. Falls back to GPU shader via `GPUChidoriRenderer` when ModernGL is available.
- **Sharingan Eye Compositor** — Overlays Sharingan/Mangekyo eye textures onto tracked iris positions with eyelid masking.

### Activation State Machine (Phase 3)

Progression through eye-contact-driven states:

```
IDLE → FACE_DETECTED → TRACKING → STARE_DETECTED → ACTIVATING → POWER_ACTIVE → COOLDOWN
                                         ↓
                                    INTERRUPTED
```

The activation engine fires events (`on_stare_start`, `on_activation_start`, `on_activation_complete`, `on_interrupt`, `on_cooldown_start/end`) that downstream systems (effects, presentation, audio, recording, intelligence) subscribe to via an event dispatcher.

### GPU Rendering (Optional)

When ModernGL is installed, `GPUChidoriRenderer` provides a full-screen fragment shader for the Chidori effect:

- **Procedural FBM noise** for turbulent plasma
- **Recursive lightning bolt function** using jagged displacement
- **Multiple directional bolts** along the hand blade direction
- **Electric arc noise** and bloom-like saturation
- Falls back to CPU rendering seamlessly if unavailable

---

## Installation

### Prerequisites

- **Python** 3.10 or higher
- **Webcam** (built-in or external USB)
- **Windows** recommended (MSMF/DirectShow backends). Linux/macOS supported via `CAP_ANY`.

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd sharingan

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.10.0.84 | Camera capture, frame processing, rendering |
| `numpy` | ≥2.0, <3.0 | Array operations, math |
| `mediapipe` | 0.10.35 | Face mesh (468 landmarks) and hand tracking (21 landmarks) |
| `pygame` | 2.6.1 | Audio playback engine *(optional, Python <3.14)* |
| `moderngl` | 5.10.0 | GPU shader rendering for Chidori *(optional, Python <3.14)* |

> **Note:** `pygame` and `moderngl` are optional extras. The engine falls back gracefully when they are unavailable — core VFX and tracking work without them.

### First-Run Model Download

On first launch, the hand landmarker model (`hand_landmarker.task`, ~10 MB) is automatically downloaded from Google's MediaPipe model repository to `~/.mediapipe/models/`. This requires an internet connection on first run only.

---

## Usage

```bash
python main.py
```

The engine opens a window titled **「 SHARINGAN CINEMA ENGINE 」** showing your webcam feed with real-time tracking and effects.

### Quick Workflow

1. Launch the engine with `python main.py`.
2. Look at the camera — the face/iris tracker locks on.
3. Hold a sustained stare → activation sequence begins (Sharingan buildup).
4. Raise your **left hand** (open palm) → **Rasengan** appears.
5. Raise your **right hand** → **Chidori** lightning activates.
6. Raise **both hands** → **Fusion mode** (both effects simultaneously).
7. Lower your hands → effects deactivate.

---

## Controls

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `F` | Toggle fullscreen |
| `D` | Toggle debug overlay (FPS, states, timing) |
| `T` | Toggle tracking debug visualization (landmarks, iris) |
| `R` | Start / stop video recording |
| `P` | Take a screenshot |
| `L` | Capture and export a replay clip |
| `C` | Toggle creator mode (hides all debug overlays for clean footage) |
| `1` | Set recording quality: LOW |
| `2` | Set recording quality: MEDIUM |
| `3` | Set recording quality: HIGH |
| `4` | Set recording quality: CINEMATIC |
| `N` | Cycle AI personality mode |
| `M` | Force-trigger a rare cinematic event |
| `5` | Personality: Calm Observer |
| `6` | Personality: Corrupted Entity |
| `7` | Personality: Unstable Power |
| `8` | Personality: Aggressive Awakening |
| `9` | Personality: Silent Void |

---

## Gesture System

Power assignment is **handedness-based** — the engine uses MediaPipe's hand classification, not gesture shapes:

| Hand | Effect | Visual |
|---|---|---|
| **Left hand** visible | **Rasengan** | Blue-white spinning energy orb anchored to palm center |
| **Right hand** visible | **Chidori** | Dense lightning blade covering the entire hand with branching arcs |
| **Both hands** visible | **Fusion** | Both effects render simultaneously |
| **No hands** visible | **Deactivate** | All hand effects clear |

> MediaPipe's handedness labels are mirrored — in the camera view, what appears as your right hand is classified as "left" by MediaPipe, which is already accounted for in the engine's mapping.

### Rasengan Details

- Layered blue-white orb with soft environmental light spill
- Orbiting spiral wisps and inner thread animation
- Bright center with pulsing burst ring
- Scales to approximately 8.5% of the smaller frame dimension

### Chidori Details

- **White-hot core** with pulsing blue plasma shell
- **Full-hand electricity**: lightning arcs crawl along every finger chain (MCP → PIP → DIP → TIP)
- **Cross-finger arcs**: chaotic connections between fingertips and palm joints
- **Violent branching arcs**: 3–4 level recursive lightning bolts shooting outward
- **Particle system**: 600-particle pool with physics-based attraction to core
- **Environmental illumination**: convex hull glow over the entire hand
- **Screen effects**: exposure boost and random horizontal shake during high intensity

---

## Configuration

All settings are centralized in `config/settings.py` using Python dataclasses. Modify `AppConfig.default()` or create a custom config:

```python
from config.settings import AppConfig, CameraConfig, EffectsConfig

config = AppConfig.default()
config.camera.width = 1920           # Full HD
config.camera.height = 1080
config.camera.fps = 60
config.effects.enable_eye_overlay = True
config.effects.enable_genjutsu_glitch = True
config.intelligence.initial_personality = "CORRUPTED_ENTITY"
```

### Key Configuration Sections

| Section | Parameters | Description |
|---|---|---|
| `CameraConfig` | `device_id`, `width`, `height`, `fps`, `backend` | Webcam capture settings |
| `TrackingConfig` | `blink_threshold`, `smoothing_factor`, `gaze_center_threshold` | Face/iris tracking tuning |
| `ActivationConfig` | `stare_duration_ms`, `activation_duration_ms`, `cooldown_duration_ms` | Eye activation timing |
| `EffectsConfig` | `enable_eye_overlay`, `enable_genjutsu_glitch`, `glitch_intensity` | VFX toggles and intensity |
| `PresentationConfig` | `hud_intensity`, `cinematic_zoom_strength`, `corruption_intensity` | Cinematic layer tuning |
| `AudioConfig` | `master_volume`, `ambience_volume`, `backend` | Audio mixing |
| `RecordingConfig` | `recording_fps`, `replay_duration_s`, `quality_mode` | Recording and export |
| `IntelligenceConfig` | `initial_personality`, `auto_adapt`, `rare_event_probability` | AI behavior tuning |

---

## Project Structure

```
sharingan/
├── main.py                          # Application entry point & SharinganEngine orchestrator
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── config/                          # Configuration system
│   └── settings.py                  # All dataclass configs (Camera, Tracking, Effects, etc.)
│
├── core/                            # Core engine (Phase 1)
│   ├── camera.py                    # CameraEngine — multi-backend webcam capture with fallback
│   └── fps_manager.py               # FPSManager — smoothed FPS counter and frame rate limiter
│
├── tracking/                        # Face & iris tracking (Phase 2)
│   ├── tracking_engine.py           # TrackingEngine — orchestrates all tracking subsystems
│   ├── tracking_data.py             # Data structures (TrackingFrameData, FaceData, EyeData, etc.)
│   ├── face_tracker.py              # MediaPipe FaceMesh integration
│   ├── iris_tracker.py              # Iris position & eye openness detection
│   ├── blink_detector.py            # Blink detection from eye aspect ratio
│   ├── eye_contact_detector.py      # Gaze-based eye contact detection
│   ├── smoothing.py                 # Exponential moving average smoothers
│   ├── debug_renderer.py            # Tracking visualization overlay
│   └── hand_tracking_state.py       # Hand tracking state container
│
├── activation/                      # Activation state machine (Phase 3)
│   ├── activation_engine.py         # Central activation controller with event dispatch
│   ├── activation_data.py           # ActivationSnapshot data structure
│   ├── activation_state.py          # ActivationState enum
│   ├── state_machine.py             # Generic state machine with transition history
│   ├── event_dispatcher.py          # Pub/sub event system
│   ├── cooldown_manager.py          # Cooldown timer
│   ├── activation_timer.py          # Elapsed time tracker
│   └── transition_manager.py        # Debounced transition logic
│
├── effects/                         # Visual effects (Phase 4)
│   ├── effect_manager.py            # EffectManager — compositor-based effect orchestrator
│   ├── anime_power_effect.py        # AnimePowerEffect — hand power assignment and rendering
│   ├── hand_power_detector.py       # MediaPipe Hands integration with gesture classification
│   ├── rasengan_effect.py           # Rasengan energy orb renderer
│   ├── chidori_effect.py            # Chidori lightning blade renderer (CPU)
│   ├── gpu_chidori.py               # GPUChidoriRenderer — ModernGL shader implementation
│   ├── simple_hand_tracker.py       # OpenCV-based hand tracker fallback (skin detection)
│   ├── base_effect.py               # BaseEffect abstract class
│   ├── compositor.py                # Effect compositor (priority-ordered rendering)
│   ├── effect_data.py               # EffectContext and EffectDebugState
│   ├── ai_eye_compositor.py         # AI-based eye compositing
│   ├── realistic_eye_compositor.py  # Realistic Sharingan eye overlay
│   ├── professional_eye_compositor.py # Production eye compositor
│   ├── gpu_eye_compositor.py        # GPU-accelerated eye compositing
│   ├── texture_eye_compositor.py    # Texture-based eye compositor
│   ├── cinematic_eye_compositor.py  # Cinematic eye compositor
│   ├── vfx_eye_effect.py            # VFX eye rendering
│   ├── eye_overlay.py               # Eye overlay system
│   ├── eye_texture_loader.py        # Sharingan texture loader
│   ├── activation_impact.py         # Screen impact on activation
│   ├── cinematic_activation_system.py # Cinematic activation sequence
│   ├── color_grading.py             # Color grading / bloom effect
│   ├── genjutsu_glitch.py           # Genjutsu glitch distortion
│   ├── screen_fx.py                 # Screen-wide post effects
│   └── world_distortion.py          # World distortion effect
│
├── presentation/                    # Cinematic presentation (Phase 5)
│   ├── presentation_manager.py      # PresentationManager — cinematic layer orchestrator
│   ├── presentation_data.py         # PresentationContext and PresentationSnapshot
│   ├── activation_sequence.py       # Stage-based activation sequence controller
│   ├── cinematic_camera.py          # Zoom and camera motion effects
│   ├── cinematic_text.py            # Animated text cue system
│   ├── corruption_system.py         # Visual corruption overlays
│   ├── hud_manager.py               # HUD overlay rendering
│   ├── immersion_layer.py           # Immersion effects (vignette, etc.)
│   └── transition_engine.py         # Transition queue and interpolation
│
├── audio/                           # Audio system (Phase 6)
│   ├── audio_manager.py             # AudioManager facade
│   ├── audio_engine.py              # Core audio engine (Pygame-based)
│   ├── audio_data.py                # AudioSnapshot data structures
│   ├── activation_audio.py          # Activation-synced sound cues
│   ├── ambience_system.py           # Ambient soundscape
│   ├── glitch_audio.py              # Glitch / distortion audio
│   ├── sound_player.py              # Low-level sound playback
│   ├── audio_sync.py                # Audio-visual synchronization
│   └── audio_transition.py          # Audio crossfade transitions
│
├── recording/                       # Recording & export (Phase 7)
│   ├── recording_manager.py         # RecordingManager facade
│   ├── recording_data.py            # RecordingState, RecordingStats, ReplayClip
│   ├── video_recorder.py            # OpenCV VideoWriter with queue-based writing
│   ├── replay_buffer.py             # Circular frame buffer for instant replay
│   ├── screenshot_manager.py        # PNG screenshot capture
│   ├── quality_manager.py           # Quality presets (LOW / MEDIUM / HIGH / CINEMATIC)
│   ├── export_manager.py            # Output directory and path management
│   └── performance_monitor.py       # Recording performance tracking
│
├── intelligence/                    # Cinematic AI (Phase 8)
│   ├── cinematic_director.py        # CinematicDirector — top-level intelligence facade
│   ├── intelligence_data.py         # DirectorSnapshot, EscalationTier
│   ├── personality_engine.py        # 5 personality modes with auto-adaptation
│   ├── behavior_tracker.py          # User behavior metrics (stare, blink, engagement)
│   ├── escalation_system.py         # Tension escalation with rise/decay rates
│   ├── rare_event_manager.py        # Probabilistic rare cinematic events
│   ├── memory_system.py             # Session memory for adaptive behavior
│   └── immersion_controller.py      # Immersion intensity control
│
├── ui/                              # User interface
│   ├── keyboard.py                  # KeyboardHandler — key bindings and callbacks
│   ├── overlay.py                   # DebugOverlay — FPS, state, and debug info
│   └── window.py                    # WindowManager — OpenCV window with fullscreen
│
├── utils/                           # Utilities
│   ├── logger.py                    # Color-coded logging with component tags
│   ├── processing.py                # FrameProcessor — flip, enhance, ensure visible
│   └── webcam_enhancer.py           # Webcam quality enhancement utilities
│
├── assets/                          # Asset files
│   ├── audio/                       # Sound effects and ambient tracks
│   │   ├── activation/              # Activation sound cues
│   │   ├── ambience/                # Background ambient audio
│   │   ├── glitch/                  # Glitch effect sounds
│   │   ├── transitions/             # Transition audio
│   │   └── ui/                      # UI interaction sounds
│   └── Sharingan Pack/              # Sharingan eye textures
│       ├── mangekyo/                # Mangekyo Sharingan variants
│       └── realistic_eyes/          # Realistic eye overlays
│
└── recordings/                      # Output directory for recordings and screenshots
    ├── sessions/                    # Recorded video sessions
    ├── replays/                     # Instant replay exports
    ├── screenshots/                 # Screenshot captures
    └── exports/                     # General exports
```

---

## Performance Optimization

### Recommended Settings

| Setting | Recommended | Notes |
|---|---|---|
| Resolution | 1280×720 | Best balance of quality and performance |
| FPS Target | 30 | Sufficient for real-time VFX; use 60 for smoother display |
| FPS Cap | 60 | Prevents unnecessary CPU usage |
| Webcam | External USB | Higher quality and more reliable than built-in laptop cameras |
| Lighting | Well-lit, front-facing | Critical for reliable hand and face tracking |
| Background | Moderate contrast | Avoid hands-matching backgrounds for better skin segmentation |

### Performance Tips

- **Disable unused effects** in `config/settings.py` — each effect (genjutsu glitch, world distortion, color grading) adds per-frame cost.
- **Use creator mode** (`C` key) to suppress debug overlays during recording for cleaner output and marginally better performance.
- **ModernGL GPU rendering** offloads Chidori shader work to the GPU. Install `moderngl` for best Chidori performance.
- **Lower resolution** (e.g., 640×480) if FPS drops below target on older hardware.
- **Close other camera apps** (Zoom, Teams, browser video) before launching — only one app can hold the camera.

### Typical Resource Usage

| Metric | Value |
|---|---|
| FPS | 30–60 (configurable) |
| CPU | 20–40% (single core, depends on effects) |
| Memory | 300–500 MB |
| GPU | Minimal unless ModernGL is active |
| Latency | <50ms capture-to-display |

---

## Troubleshooting

### Camera Issues

| Problem | Solution |
|---|---|
| "Failed to open camera" | Close all other apps using the camera (Zoom, Teams, browser video calls). |
| Black or frozen frame | Try changing `backend` in `CameraConfig` to `"dshow"` or `"msmf"` (Windows). |
| Wrong camera selected | Change `device_id` in `CameraConfig` (try 0, 1, 2). |
| Low FPS from camera | Some USB cameras default to low FPS. Ensure the camera supports 30 FPS at your target resolution. |

### MediaPipe Issues

| Problem | Solution |
|---|---|
| Hand model download fails | Ensure internet access on first run. Model is cached at `~/.mediapipe/models/`. |
| Face not detected | Ensure your face is well-lit, facing the camera, and within the frame. |
| Hand tracking unreliable | Improve lighting. Avoid backgrounds with skin-like colors. Move hands closer to camera. |
| "No module named mediapipe" | Run `pip install mediapipe==0.10.35`. |

### GPU / ModernGL Issues

| Problem | Solution |
|---|---|
| ModernGL not available | This is optional. The engine falls back to CPU rendering automatically. |
| Shader compilation error | Update GPU drivers. ModernGL requires OpenGL 3.3+ support. |
| Import error on Python 3.14+ | `moderngl` is excluded for Python 3.14+. Use CPU rendering or an earlier Python version. |

### General Issues

| Problem | Solution |
|---|---|
| No audio | Install `pygame` (`pip install pygame==2.6.1`). Audio is optional. |
| Effects not appearing | Ensure `effects.enabled = True` in config. Check that hands are detected (press `D` for debug). |
| High CPU usage | Reduce resolution, disable unused effects, cap FPS lower. |
| Recording not working | Check that `recordings/` directory exists and is writable. |

---

## AI Personality System

The Cinematic Director (Phase 8) adapts the engine's behavior based on user interaction patterns:

| Personality | Baseline Tension | Description |
|---|---|---|
| **Calm Observer** | Low | Subtle, ambient pacing with slow escalation |
| **Corrupted Entity** | Medium | Glitchy, unstable visual corruption |
| **Unstable Power** | High | Rapid escalation, volatile effects |
| **Aggressive Awakening** | Very High | Intense, explosive cinematic moments |
| **Silent Void** | Minimal | Quiet, eerie calm with sudden bursts |

The system tracks user behavior (stare duration, blink frequency, engagement) and can auto-adapt personality based on interaction patterns when `auto_adapt` is enabled.

---

## Future Improvements

- [ ] Additional jutsu effects (Amaterasu black flames, Susanoo aura)
- [ ] Full-body pose tracking for extended VFX coverage
- [ ] Web-based configuration UI
- [ ] Multi-camera support
- [ ] Network streaming output (RTMP/NDI)
- [ ] Plugin system for community effects
- [ ] Mobile companion app for remote control
- [ ] Persistent user profiles with session history

---

## License

This project is for educational and personal use. Naruto and related character names are trademarks of their respective owners.
