"""Compatibility import; implementation lives in backend/app/knowledge_base."""
import sys
from pathlib import Path
_BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
from app.knowledge_base.service import *  # noqa: F401,F403,E402
