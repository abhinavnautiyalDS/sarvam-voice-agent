"""Compatibility entrypoint for existing deployments using ``uvicorn api:app``."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.main import app  # noqa: E402

__all__ = ["app"]
