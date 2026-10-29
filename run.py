"""
ChronoVideo Runner
Entrypoint script to start the ChronoVideo platform.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Link to pre-installed venv packages (PyMuPDF, imageio_ffmpeg, groq, etc.)
VENV_PKGS = r"C:\Users\Admin\Downloads\LectureLens_skeleton\lecturelens\backend\venv\Lib\site-packages"
if os.path.exists(VENV_PKGS) and VENV_PKGS not in sys.path:
    sys.path.append(VENV_PKGS)

if __name__ == "__main__":
    from backend.app.main import run
    run()
