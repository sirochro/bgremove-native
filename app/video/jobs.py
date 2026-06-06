"""Video matting jobs: decode → RVM infer → alpha post-process → encode."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion

from app.video.matting import RVMSession, suggest_downsample_ratio
from app.video.pipeline import (
    iter_frames,
    open_decoder,
    open_prores_encoder,
    open_webm_encoder,
    probe,
)


def postprocess_alpha(
    pha: np.ndarray,
    low: int,
    high: int,
    edge: int,
    feather: int,
) -> np.ndarray:
    """Apply low cut → high boost → expand/contract → Gaussian feather."""
    out = pha
    if low > 0:
        out = np.where(out < low, np.uint8(0), out)
    if high < 255:
        out = np.where(out > high, np.uint8(255), out)
    if edge > 0:
        size = 2 * edge + 1
        out = grey_dilation(out, size=(size, size))
    elif edge < 0:
        size = -2 * edge + 1
        out = grey_erosion(out, size=(size, size))
    if feather > 0:
        out = gaussian_filter(out, sigma=feather / 2.0)
    return out.astype(np.uint8)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Job:
    id: str
    input_path: str
    output_path: str
    model_path: str
    output_format: str  # "prores" or "webm"
    downsample_ratio: Optional[float] = None

    # Alpha post-processing
    alpha_low: int = 0
    alpha_high: int = 255
    edge_expand: int = 0
    feather: int = 0

    mute: bool = False

    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    fps_processed: float = 0.0
    n_frames_total: int = 0
    n_frames_done: int = 0
    eta_seconds: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        output_format: str,
        downsample_ratio: Optional[float] = None,
        alpha_low: int = 0,
        alpha_high: int = 255,
        edge_expand: int = 0,
        feather: int = 0,
        mute: bool = False,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:8],
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            output_format=output_format,
            downsample_ratio=downsample_ratio,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            edge_expand=edge_expand,
            feather=feather,
            mute=mute,
        )
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job._cancel.set()
        return True

    def _run(self, job: Job) -> None:
        decoder = None
        encoder = None
        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            info = probe(job.input_path)
            W, H, fps = info["width"], info["height"], info["fps"]
            job.n_frames_total = info["n_frames"]

            dsr = job.downsample_ratio
            if dsr is None:
                dsr = suggest_downsample_ratio(W, H)

            session = RVMSession(job.model_path)
            job.provider = session.active_provider

            audio_source = None if job.mute else job.input_path

            decoder = open_decoder(job.input_path)
            if job.output_format == "webm":
                encoder = open_webm_encoder(
                    job.output_path, W, H, fps, audio_source=audio_source
                )
            else:
                encoder = open_prores_encoder(
                    job.output_path, W, H, fps, audio_source=audio_source
                )

            assert encoder.stdin is not None

            low = max(0, min(255, int(job.alpha_low)))
            high = max(0, min(255, int(job.alpha_high)))
            edge = max(-5, min(5, int(job.edge_expand)))
            feather = max(0, min(20, int(job.feather)))
            do_post = low > 0 or high < 255 or edge != 0 or feather > 0

            t0 = time.time()
            for idx, rgb in enumerate(iter_frames(decoder, W, H)):
                if job._cancel.is_set():
                    job.status = JobStatus.CANCELED
                    return

                fgr, pha = session.infer(rgb, dsr)
                if do_post:
                    pha = postprocess_alpha(pha, low, high, edge, feather)
                rgba = np.dstack([fgr, pha])
                encoder.stdin.write(rgba.tobytes())

                job.n_frames_done = idx + 1
                if job.n_frames_total > 0:
                    job.progress = min(1.0, job.n_frames_done / job.n_frames_total)
                dt = time.time() - t0
                if dt > 0:
                    job.fps_processed = job.n_frames_done / dt
                    remaining = max(0, job.n_frames_total - job.n_frames_done)
                    if job.fps_processed > 0:
                        job.eta_seconds = remaining / job.fps_processed

            encoder.stdin.close()
            encoder.wait(timeout=900)
            decoder.wait(timeout=60)

            job.progress = 1.0
            job.status = JobStatus.DONE
            job.finished_at = time.time()
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.FAILED
            job.finished_at = time.time()
        finally:
            for proc in (decoder, encoder):
                if proc is not None and proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
