"""Video matting API routes (mounted under /api/video)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.video.jobs import Job, JobManager, JobStatus
from app.video.pipeline import FFMPEG, FFPROBE

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

router = APIRouter()
_manager = JobManager()


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
