"""
Eye texture loader for Sharingan Pack assets.
Loads and processes full anime eye textures with proper extraction.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class EyeTexture:
    """Single eye texture with metadata."""

    def __init__(
        self, name: str, texture: np.ndarray, mask: Optional[np.ndarray] = None
    ):
        self.name = name
        self.texture = texture
        self.mask = mask
        self.height, self.width = texture.shape[:2]


class SharinganTextureLoader:
    """Loads and manages Sharingan Pack eye textures."""

    def __init__(self, pack_path: Optional[str] = None):
        # Defer to centralized config when not provided explicitly
        if pack_path is None:
            from config.assets import get_sharingan_pack_path

            pack_path = get_sharingan_pack_path()

        self.pack_path = Path(pack_path)
        self.textures: List[EyeTexture] = []
        self._current_index = 0
        self._load_textures()

    def _load_textures(self) -> None:
        """Load all eye textures from Sharingan Pack."""
        if not self.pack_path.exists():
            return

        renders_dir = self.pack_path / "Renders"
        if renders_dir.exists():
            for img_path in renders_dir.glob("*.png"):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.shape[2] == 4:
                        texture, mask = self._extract_eye_regions(img)
                        if texture is not None:
                            self.textures.append(
                                EyeTexture(
                                    name=img_path.stem, texture=texture, mask=mask
                                )
                            )
                except Exception:
                    pass

        main_path = self.pack_path / "MAIN.png"
        if main_path.exists():
            try:
                img = cv2.imread(str(main_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    textures = self._extract_eyes_from_chart(img)
                    self.textures.extend(textures)
            except Exception:
                pass

        for lvl in range(1, 4):
            lvl_path = self.pack_path / f"Sharingan LVL {lvl}.png"
            if lvl_path.exists():
                try:
                    img = cv2.imread(str(lvl_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        textures = self._extract_eyes_from_chart(img)
                        self.textures.extend(textures)
                except Exception:
                    pass

    def _extract_eye_regions(
        self, img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract eye region from single eye image."""
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            if alpha is None:
                return None, None
            valid_coords = np.where(alpha > 20)
            if len(valid_coords[0]) == 0:
                return None, None
            y_min, y_max = valid_coords[0].min(), valid_coords[0].max()
            x_min, x_max = valid_coords[1].min(), valid_coords[1].max()
            eye = img[y_min:y_max, x_min:x_max]
            mask = alpha[y_min:y_max, x_min:x_max]
            return eye, mask
        return img, None

    def _extract_eyes_from_chart(self, img: np.ndarray) -> List[EyeTexture]:
        """Extract individual eyes from chart image."""
        textures = []
        h, w = img.shape[:2]

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        eyes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000 and area < (h * w * 0.3):
                x, y, cw, ch = cv2.boundingRect(cnt)
                if ch > 0 and cw > 0:
                    aspect = cw / ch
                    if 0.5 < aspect < 2.5:
                        eyes.append((x, y, cw, ch))

        eyes = sorted(eyes, key=lambda e: e[0])

        for i, (x, y, cw, ch) in enumerate(eyes[:6]):
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)

            eye_img = img[y1:y2, x1:x2]
            if eye_img.size > 0:
                textures.append(EyeTexture(name=f"chart_eye_{i}", texture=eye_img))

        return textures

    def get_random_texture(self) -> Optional[EyeTexture]:
        """Get random eye texture."""
        if not self.textures:
            return None
        return random.choice(self.textures)

    def get_next_texture(self) -> Optional[EyeTexture]:
        """Get next texture in sequence."""
        if not self.textures:
            return None
        texture = self.textures[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.textures)
        return texture

    def get_texture_by_name(self, name: str) -> Optional[EyeTexture]:
        """Get specific texture by name."""
        for tex in self.textures:
            if name.lower() in tex.name.lower():
                return tex
        return None

    def count(self) -> int:
        """Return number of loaded textures."""
        return len(self.textures)


def create_texture_loader(pack_path: Optional[str] = None) -> SharinganTextureLoader:
    """Factory function to create texture loader."""
    return SharinganTextureLoader(pack_path)
