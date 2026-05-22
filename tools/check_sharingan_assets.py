import sys
from pathlib import Path

# Ensure project root is on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from config.assets import get_sharingan_pack_path
from effects.eye_texture_loader import create_texture_loader

p = get_sharingan_pack_path()
print('SHARINGAN_PACK_PATH:', p)
print('Exists:', Path(p).exists())
print('Contents:', [str(x) for x in Path(p).iterdir()])

loader = create_texture_loader()
print('Loaded texture count:', loader.count())

# Try to load realistic compositor textures
from effects.realistic_eye_compositor import ProfessionalEyeCompositor

comp = ProfessionalEyeCompositor()
print('Realistic compositor textures:', len(getattr(comp, 'textures', [])))
