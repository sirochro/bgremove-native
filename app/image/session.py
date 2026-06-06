"""rembg DirectML session pool for image background removal."""
from __future__ import annotations

from typing import Dict

from fastapi import HTTPException
from rembg import new_session
from rembg.sessions.base import BaseSession

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

_sessions: Dict[str, BaseSession] = {}


def get_session(model_name: str) -> BaseSession:
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _sessions:
        _sessions[model_name] = new_session(model_name, providers=PROVIDERS)
    return _sessions[model_name]


def loaded_models() -> list[str]:
    return list(_sessions.keys())
