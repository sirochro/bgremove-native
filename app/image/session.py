"""rembg DirectML session cache for image background removal.

Only one session is held at a time. See app/session_cache.py for the
VRAM-exhaustion rationale that made this necessary.
"""
from __future__ import annotations

from fastapi import HTTPException
from rembg import new_session
from rembg.sessions.base import BaseSession

from app.session_cache import SingleSessionCache

AVAILABLE_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "sam",
]

DEFAULT_MODEL = "u2net"
PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


def _load(model_name: str) -> BaseSession:
    return new_session(model_name, providers=PROVIDERS)


_cache: SingleSessionCache[BaseSession] = SingleSessionCache(_load)


def get_session(model_name: str) -> BaseSession:
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    return _cache.get(model_name)


def loaded_models() -> list[str]:
    return _cache.loaded()
