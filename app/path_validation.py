"""Path validation for input/output file APIs.

Local-only FastAPI doesn't mean "safe to accept any path". The
pywebview process listens on 127.0.0.1, but other local processes,
browsers via DNS rebinding, and stray automation can still reach it.
Treat every path as untrusted: enforce an allow-list of extensions,
a deny-list of system directories, and a hard cap on input file size.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException


INPUT_IMAGE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
)
INPUT_VIDEO_EXTS = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
)
OUTPUT_IMAGE_EXTS = frozenset({".png"})
OUTPUT_VIDEO_EXTS = frozenset({".mov", ".webm"})

MAX_INPUT_IMAGE_BYTES = 200 * 1024 * 1024            # 200 MB
MAX_INPUT_VIDEO_BYTES = 5 * 1024 * 1024 * 1024       # 5 GB


def _system_deny_prefixes() -> list[Path]:
    """System paths the app must never read from or write to.

    Resolved at first use; cached. Resolution follows symlinks so a
    symlink that masquerades as a user path can't aim a write at, say,
    ``/etc``.
    """
    prefixes: list[Path] = []
    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        prefixes.append(Path(sysroot))
        for env in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            p = os.environ.get(env)
            if p:
                prefixes.append(Path(p))
        appdata = os.environ.get("APPDATA")
        if appdata:
            prefixes.append(Path(appdata) / "Microsoft")
    else:
        prefixes += [
            Path("/etc"),
            Path("/root"),
            Path("/sys"),
            Path("/proc"),
            Path("/dev"),
            Path.home() / ".ssh",
            Path.home() / ".config",
        ]
    resolved: list[Path] = []
    for p in prefixes:
        try:
            if p.exists():
                resolved.append(p.resolve())
        except OSError:
            continue
    return resolved


_DENY_CACHE: list[Path] | None = None


def _deny_prefixes() -> list[Path]:
    global _DENY_CACHE
    if _DENY_CACHE is None:
        _DENY_CACHE = _system_deny_prefixes()
    return _DENY_CACHE


def _is_under(target: Path, prefix: Path) -> bool:
    try:
        target.relative_to(prefix)
        return True
    except ValueError:
        return False


def _resolve_absolute(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        raise HTTPException(400, f"Path must be absolute: {path}")
    try:
        return p.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, f"Could not resolve path: {path} ({exc})") from exc


def _check_not_in_deny(resolved: Path) -> None:
    for prefix in _deny_prefixes():
        if _is_under(resolved, prefix):
            raise HTTPException(
                400, f"Path is inside a protected system directory: {resolved}"
            )


def _check_extension(resolved: Path, allowed: Iterable[str]) -> None:
    if resolved.suffix.lower() not in allowed:
        raise HTTPException(400, f"Unsupported file extension: {resolved.suffix}")


def validate_input_path(path: str, kind: str) -> Path:
    """Validate an input file path. ``kind`` ∈ {'image', 'video'}.

    Returns the resolved (symlink-followed) absolute Path on success.
    """
    if kind == "image":
        exts = INPUT_IMAGE_EXTS
        max_bytes = MAX_INPUT_IMAGE_BYTES
    elif kind == "video":
        exts = INPUT_VIDEO_EXTS
        max_bytes = MAX_INPUT_VIDEO_BYTES
    else:
        raise HTTPException(500, f"Unknown input kind: {kind}")

    resolved = _resolve_absolute(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(400, f"Input file not found: {path}")
    _check_not_in_deny(resolved)
    _check_extension(resolved, exts)

    size = resolved.stat().st_size
    if size > max_bytes:
        raise HTTPException(
            400, f"Input file too large: {size} bytes (max {max_bytes})"
        )

    return resolved


def validate_output_path(path: str, kind: str) -> Path:
    """Validate an output file path. ``kind`` ∈ {'image', 'video'}.

    Parent directory must already exist — the API will NOT mkdir parents
    on the user's behalf because that lets an attacker scaffold arbitrary
    directory trees on the filesystem.
    """
    if kind == "image":
        exts = OUTPUT_IMAGE_EXTS
    elif kind == "video":
        exts = OUTPUT_VIDEO_EXTS
    else:
        raise HTTPException(500, f"Unknown output kind: {kind}")

    resolved = _resolve_absolute(path)
    _check_not_in_deny(resolved)
    _check_extension(resolved, exts)

    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(
            400, f"Output directory does not exist: {parent}"
        )

    return resolved
