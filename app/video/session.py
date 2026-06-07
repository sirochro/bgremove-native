"""Shared RVM session cache for video preview + jobs.

Without sharing, preview holds one session for its current model and
each job allocates a second one for the same model, so up to two
multi-hundred-MB DirectML sessions can sit in VRAM at the same time.
Routing both through a single SingleSessionCache caps the maximum at
one. Preview and the active job compete for that slot — switching
models pays a re-load, exactly as in the image path.

Callers are expected to call ``session.reset()`` if they need fresh
recurrent state (job clips are independent, so the job loop does this
at the top of every run).
"""
from __future__ import annotations

from app.session_cache import SingleSessionCache
from app.video.matting import RVMSession

_cache: SingleSessionCache[RVMSession] = SingleSessionCache(
    lambda path: RVMSession(path)
)


def get_session(model_path: str) -> RVMSession:
    return _cache.get(model_path)


def loaded() -> list[str]:
    return _cache.loaded()
