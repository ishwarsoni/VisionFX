# VisionFX

Real-time computer vision and AR/VFX experiments built with Python, OpenCV, and MediaPipe.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/opencv-4.10-green?style=flat-square" alt="OpenCV">
  <img src="https://img.shields.io/badge/mediapipe-0.10-orange?style=flat-square" alt="MediaPipe">
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" alt="License">
</p>

## Overview

VisionFX is an experimental real-time computer vision and AR/VFX project. It uses webcam input, MediaPipe tracking, and OpenCV rendering to drive hand tracking, gesture detection, anime-inspired visual effects, and low-latency overlays.

## Features

- Real-time webcam-based hand and face tracking
- Gesture-driven visual effects and anime-inspired overlays
- Low-latency frame processing with OpenCV
- Optional audio, recording, and presentation layers
- Configurable pipeline with GPU-accelerated options where available

## Demo

<video controls width="640">
  <source src="Demo%20Video/VN20260522_135217.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the demo: [Demo Video/VN20260522_135217.mp4](Demo%20Video/VN20260522_135217.mp4)
</video>

Add additional screenshots or GIFs below if desired.

## Installation

```bash
git clone https://github.com/ishwarsoni/VisionFX.git
cd VisionFX
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The application opens a webcam window and starts the tracking and VFX pipeline.

## Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `F` | Toggle fullscreen |
| `D` | Toggle debug overlay |
| `T` | Toggle tracking debug visualization |
| `R` | Start or stop recording |
| `P` | Take a screenshot |
| `L` | Capture and export a replay clip |
| `C` | Toggle creator mode |
| `1` | Recording quality: LOW |
| `2` | Recording quality: MEDIUM |
| `3` | Recording quality: HIGH |
| `4` | Recording quality: CINEMATIC |
| `N` | Cycle AI personality mode |
| `M` | Trigger a rare cinematic event |
| `5` | Personality: Calm Observer |
| `6` | Personality: Corrupted Entity |
| `7` | Personality: Unstable Power |
| `8` | Personality: Aggressive Awakening |
| `9` | Personality: Silent Void |

## Tech Stack

| Layer | Tools |
|---|---|
| Core runtime | Python |
| Vision and rendering | OpenCV, NumPy |
| Tracking | MediaPipe |
| Audio | Pygame (optional) |
| GPU effects | ModernGL (optional) |

## Project Structure

```text
VisionFX/
├── main.py
├── config/
├── core/
├── tracking/
├── activation/
├── effects/
├── presentation/
├── audio/
├── recording/
├── intelligence/
├── ui/
├── utils/
├── assets/
│   ├── audio/
│   └── sharingan_pack/
├── tools/
├── docs/
├── requirements.txt
└── LICENSE
```

## Performance Notes

- `1280x720` at `30 FPS` is a practical default for most systems.
- Lower the resolution or disable extra effects if frame rate drops.
- Optional GPU features depend on ModernGL and a compatible OpenGL setup.
- The project is designed to fall back gracefully when optional dependencies are unavailable.

## Future Improvements

- Additional visual effects and overlays
- More configuration options for recording and export
- Better demo assets and example captures
- Optional web-based control panel

## License

MIT License. See [LICENSE](LICENSE).