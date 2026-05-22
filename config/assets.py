from pathlib import Path

# Centralized Sharingan pack location. Use this constant across the codebase
# so asset moves are safe and references are consistent.
ROOT = Path(__file__).parent.parent
SHARINGAN_PACK_DIR = ROOT / "assets" / "sharingan_pack"


def ensure_sharingan_pack_dir() -> Path:
    SHARINGAN_PACK_DIR.mkdir(parents=True, exist_ok=True)
    return SHARINGAN_PACK_DIR


def get_sharingan_pack_path() -> str:
    return str(ensure_sharingan_pack_dir())
