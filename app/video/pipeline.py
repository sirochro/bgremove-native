"""ffmpeg / ffprobe I/O pipelines for video matting."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = ROOT / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "bin" / "ffprobe.exe"


class ToolsMissingError(RuntimeError):
    """Raised when bundled ffmpeg/ffprobe binaries are absent."""


def ensure_tools() -> None:
    missing = [t.name for t in (FFMPEG, FFPROBE) if not t.exists()]
    if missing:
        raise ToolsMissingError(
            f"Missing binaries in bin/: {missing}. "
            "Run setup.bat or place Windows ffmpeg.exe and ffprobe.exe under bin/."
        )


def probe(path: str) -> dict:
    """Return ``{width, height, fps, n_frames, duration}``."""
    ensure_tools()
    cmd = [
        str(FFPROBE),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out on {path}") from exc
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    s = streams[0]
    fmt = data.get("format", {})

    num, denom = s["avg_frame_rate"].split("/")
    fps = float(num) / float(denom) if float(denom) != 0 else 0.0

    duration = float(s.get("duration") or fmt.get("duration") or 0.0)
    n_frames_raw = s.get("nb_frames")
    if n_frames_raw and str(n_frames_raw) != "N/A":
        n_frames = int(n_frames_raw)
    else:
        n_frames = int(round(duration * fps)) if fps > 0 else 0

    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "n_frames": n_frames,
        "duration": duration,
    }


def open_decoder(path: str) -> subprocess.Popen:
    """Decode input video → raw RGB24 frames on stdout."""
    ensure_tools()
    cmd = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )


def _open_rgba_encoder(
    out_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None,
    audio_args: list[str],
    video_args: list[str],
) -> subprocess.Popen:
    """Common rawvideo-RGBA stdin → muxed output encoder pipeline."""
    ensure_tools()
    cmd: list[str] = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
    ]
    if audio_source:
        cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a?"] + audio_args
    else:
        cmd += ["-map", "0:v:0"]
    cmd += video_args + ["-shortest", out_path]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )


def open_prores_encoder(
    out_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
) -> subprocess.Popen:
    """Encode raw RGBA from stdin → MOV ProRes 4444 with alpha (+ AAC audio)."""
    return _open_rgba_encoder(
        out_path, width, height, fps, audio_source,
        audio_args=["-c:a", "aac", "-b:a", "192k"],
        video_args=[
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-vendor", "apl0",
        ],
    )


def open_webm_encoder(
    out_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
) -> subprocess.Popen:
    """Encode raw RGBA from stdin → WebM VP9 with alpha (+ Opus audio)."""
    return _open_rgba_encoder(
        out_path, width, height, fps, audio_source,
        audio_args=["-c:a", "libopus", "-b:a", "128k"],
        video_args=[
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            "-b:v", "8M",
        ],
    )


def iter_frames(
    decoder: subprocess.Popen, width: int, height: int
) -> Iterator[np.ndarray]:
    """Yield (H, W, 3) uint8 RGB frames; reads strictly frame_size bytes per frame."""
    frame_size = width * height * 3
    assert decoder.stdout is not None
    while True:
        buf = bytearray()
        while len(buf) < frame_size:
            chunk = decoder.stdout.read(frame_size - len(buf))
            if not chunk:
                return  # EOF
            buf.extend(chunk)
        yield np.frombuffer(bytes(buf), dtype=np.uint8).reshape(height, width, 3)
