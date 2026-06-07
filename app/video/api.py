"""Video matting API routes (mounted under /api/video)."""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.video.jobs import Job, JobManager, JobStatus
from app.video.pipeline import FFMPEG, FFPROBE
from app.video.preview import preview_frame

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

router = APIRouter()
_manager = JobManager()

# Cache: result_id → PNG bytes (used for first-frame previews).
_MAX_PREVIEWS = 40
_previews: "OrderedDict[str, bytes]" = OrderedDict()


def _add_preview(data: bytes) -> str:
    pid = uuid.uuid4().hex[:12]
    _previews[pid] = data
    while len(_previews) > _MAX_PREVIEWS:
        _previews.popitem(last=False)
    return pid


def list_models() -> list[dict]:
    return [
        {
            "name": p.stem,
            "path": str(p),
            "size_mb": round(p.stat().st_size / 1_048_576, 1),
        }
        for p in sorted(MODELS_DIR.glob("rvm_*.onnx"))
    ]


@router.get("/health")
def health() -> dict:
    models = list_models()
    return {
        "ffmpeg": FFMPEG.exists(),
        "ffprobe": FFPROBE.exists(),
        "models": models,
        "ready": FFMPEG.exists() and FFPROBE.exists() and bool(models),
    }


# ---------------- First-frame preview ----------------

class PreviewRequest(BaseModel):
    input_path: str
    model: str
    alpha_low: int = 0
    alpha_high: int = 255
    edge_expand: int = 0
    feather: int = 0
    downsample_ratio: Optional[float] = None


@router.post("/preview")
def create_preview(req: PreviewRequest) -> dict:
    models = {m["name"]: m["path"] for m in list_models()}
    if req.model not in models:
        raise HTTPException(400, f"Unknown model: {req.model}")
    if not Path(req.input_path).exists():
        raise HTTPException(400, f"Input file not found: {req.input_path}")
    try:
        orig, proc, info = preview_frame(
            req.input_path,
            models[req.model],
            alpha_low=req.alpha_low,
            alpha_high=req.alpha_high,
            edge_expand=req.edge_expand,
            feather=req.feather,
            downsample_ratio=req.downsample_ratio,
        )
    except Exception as exc:
        raise HTTPException(500, f"Preview failed: {exc}") from exc
    return {
        "orig_id": _add_preview(orig),
        "proc_id": _add_preview(proc),
        "info": info,
    }


@router.get("/preview/{pid}")
def get_preview(pid: str) -> Response:
    data = _previews.get(pid)
    if data is None:
        raise HTTPException(404, "Preview not found")
    return Response(content=data, media_type="image/png")


# ---------------- Jobs (full encode) ----------------

class JobRequest(BaseModel):
    input_path: str
    output_path: str
    model: str
    output_format: str = "prores"
    downsample_ratio: Optional[float] = None
    alpha_low: int = 0
    alpha_high: int = 255
    edge_expand: int = 0
    feather: int = 0
    mute: bool = False


@router.post("/jobs")
def create_job(req: JobRequest) -> dict:
    models = {m["name"]: m["path"] for m in list_models()}
    if req.model not in models:
        raise HTTPException(400, f"Unknown model: {req.model}")
    if req.output_format not in ("prores", "webm"):
        raise HTTPException(400, f"Unknown output_format: {req.output_format}")
    if not Path(req.input_path).exists():
        raise HTTPException(400, f"Input file not found: {req.input_path}")

    job = _manager.submit(
        input_path=req.input_path,
        output_path=req.output_path,
        model_path=models[req.model],
        output_format=req.output_format,
        downsample_ratio=req.downsample_ratio,
        alpha_low=req.alpha_low,
        alpha_high=req.alpha_high,
        edge_expand=req.edge_expand,
        feather=req.feather,
        mute=req.mute,
    )
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_dict(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not _manager.cancel(job_id):
        raise HTTPException(400, "Cannot cancel job")
    return {"ok": True}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = _manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    async def event_gen():
        last_state = None
        while True:
            payload = _job_dict(job)
            if payload != last_state:
                yield {"data": json.dumps(payload)}
                last_state = payload
            if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED):
                return
            await asyncio.sleep(0.3)

    return EventSourceResponse(event_gen())


def _job_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status.value,
        "progress": job.progress,
        "fps": round(job.fps_processed, 2),
        "n_done": job.n_frames_done,
        "n_total": job.n_frames_total,
        "eta_seconds": round(job.eta_seconds, 1),
        "provider": job.provider,
        "error": job.error,
    }
