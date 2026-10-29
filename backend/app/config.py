"""
ChronoVideo - Configuration and Environment Settings
"""

import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
APP_DIR = BACKEND_DIR / "app"
DATA_DIR = BACKEND_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_storage"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Environment Keys
ENV_FILE = BACKEND_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8000))
HOST = "0.0.0.0"

# Application Metadata
APP_TITLE = "ChronoVideo API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Time-Indexed Multimodal Video Intelligence & Semantic RAG Engine"
