"""First-frame preview pipeline: ffmpeg extract → RVM infer → alpha post-process."""
from __future__ import annotations

import subprocess
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from app.alpha import clamp_params, is_default, postprocess_alpha
from app.video.matting import RVMSession, suggest_downsample_ratio
from app.video.pipeline import FFMPEG, ensure_tools, probe

# Shared session cache (reused across previews of the same model).
_sessions: dict[str, RVMSession] = {}


def get_session(model_path: str) -> RVMSession:
    if model_path not in _sessions:
        _sessions[model_path] = RVMSession(model_path)
    return _sessions[model_path]


def extract_first_frame(video_path: str, width: int, height: int) -> np.ndarray:
    """Extract the first frame as (H, W, 3) uint8 RGB via ffmpeg."""
    ensure_tools()
    cmd = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    expected = width * height * 3
    if len(out) < expected:
        raise RuntimeError(
            f"ffmpeg returned {len(out)} bytes, expected {expected} ({width}x{height})"
        )
    return np.frombuffer(out[:expected], dtype=np.uint8).reshape(height, width, 3)


def preview_frame(
    video_path: str,
    model_path: str,
    alpha_low: int = 0,
    alpha_high: int = 255,
    edge_expand: int = 0,
    feather: int = 0,
    downsample_ratio: Optional[float] = None,
) -> tuple[bytes, bytes, dict]:
    """Return ``(original_png, processed_png, info)`` for the first frame.

    ``info`` contains ``{width, height, fps, n_frames, duration}`` from ffprobe.
    """
    info = probe(video_path)
    W, H = info["width"], info["height"]
    rgb = extract_first_frame(video_path, W, H)

    orig_buf = BytesIO()
    Image.fromarray(rgb).save(orig_buf, format="PNG")

    session = get_session(model_path)
    session.reset()
    dsr = (
        downsample_ratio
        if downsample_ratio is not None
        else suggest_downsample_ratio(W, H)
    )
    fgr, pha = session.infer(rgb, dsr)

    low, high, edge, feath = clamp_params(alpha_low, alpha_high, edge_expand, feather)
    if not is_default(low, high, edge, feath):
        pha = postprocess_alpha(pha, low, high, edge, feath)

    rgba = np.dstack([fgr, pha])
    proc_buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(proc_buf, format="PNG")

    return orig_buf.getvalue(), proc_buf.getvalue(), info
