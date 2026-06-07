"""Video matting API routes (mounted under /api/video)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.blob_cache import BlobCache
from app.path_validation import validate_input_path, validate_output_path
from app.video.jobs import Job, JobManager, JobStatus
from app.video.pipeline import FFMPEG, FFPROBE
from app.video.preview import preview_frame

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

router = APIRouter()
_manager = JobManager()

# First-frame preview blobs (orig + processed PNG, one pair per request).
# Bounded by both count and total bytes (500 MB).
_previews = BlobCache(max_count=40, max_bytes=500 * 1024 * 1024)


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
    alpha_low: int = Field(0, ge=0, le=255)
    alpha_high: int = Field(255, ge=0, le=255)
    edge_expand: int = Field(0, ge=-5, le=5)
    feather: int = Field(0, ge=0, le=20)
    downsample_ratio: Optional[float] = Field(None, gt=0.0, le=1.0)


@router.post("/preview")
def create_preview(req: PreviewRequest) -> dict:
    models = {m["name"]: m["path"] for m in list_models()}
    if req.model not in models:
        raise HTTPException(400, f"Unknown model: {req.model}")
    src = validate_input_path(req.input_path, "video")
    try:
        orig, proc, info = preview_frame(
            str(src),
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
        "orig_id": _previews.add(orig),
        "proc_id": _previews.add(proc),
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
    downsample_ratio: Optional[float] = Field(None, gt=0.0, le=1.0)
    alpha_low: int = Field(0, ge=0, le=255)
    alpha_high: int = Field(255, ge=0, le=255)
    edge_expand: int = Field(0, ge=-5, le=5)
    feather: int = Field(0, ge=0, le=20)
    mute: bool = False


@router.post("/jobs")
def create_job(req: JobRequest) -> dict:
    models = {m["name"]: m["path"] for m in list_models()}
    if req.model not in models:
        raise HTTPException(400, f"Unknown model: {req.model}")
    if req.output_format not in ("prores", "webm"):
        raise HTTPException(400, f"Unknown output_format: {req.output_format}")

    src = validate_input_path(req.input_path, "video")
    out = validate_output_path(req.output_path, "video")
    # The output extension must agree with the chosen encoder.
    if req.output_format == "prores" and out.suffix.lower() != ".mov":
        raise HTTPException(400, "output_format=prores requires a .mov output path")
    if req.output_format == "webm" and out.suffix.lower() != ".webm":
        raise HTTPException(400, "output_format=webm requires a .webm output path")

    job = _manager.submit(
        input_path=str(src),
        output_path=str(out),
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
