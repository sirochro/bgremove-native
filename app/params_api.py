"""Persistent params/preview-bg store on the server side.

Avoids relying on WebView2's localStorage, which has proven flaky to
persist across pywebview launches.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body

router = APIRouter()


def _data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "bgremove-native"
    d.mkdir(parents=True, exist_ok=True)
    return d


_STATE_FILE = _data_dir() / "ui_state.json"


def _load() -> dict[str, Any]:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> None:
    tmp = _STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_STATE_FILE)


@router.get("")
def get_state() -> dict:
    return _load()


@router.post("")
def set_state(data: dict = Body(...)) -> dict:
    _save(data or {})
    return {"ok": True}
