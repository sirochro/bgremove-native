"""Video matting jobs: decode → RVM infer → alpha post-process → encode."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.alpha import clamp_params, is_default, postprocess_alpha
from app.video.matting import RVMSession, suggest_downsample_ratio
from app.video.pipeline import (
    iter_frames,
    open_decoder,
    open_prores_encoder,
    open_webm_encoder,
    probe,
)
from app.video.session import get_session


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


@dataclass
class _RunCtx:
    """All mutable handles owned by one _run invocation.

    Threaded through setup → loop → finalize → cleanup so that the
    cleanup pass can find every subprocess and thread to tear down,
    regardless of where in the pipeline an exit happened.
    """

    decoder: Optional[subprocess.Popen] = None
    encoder: Optional[subprocess.Popen] = None
    writer_t: Optional[threading.Thread] = None
    encode_q: Optional["queue.Queue[Optional[bytes]]"] = None
    writer_error: List[BaseException] = field(default_factory=list)
    session: Optional[RVMSession] = None
    W: int = 0
    H: int = 0
    fps: float = 0.0
    dsr: float = 1.0
    do_post: bool = False
    low: int = 0
    high: int = 255
    edge: int = 0
    feather: int = 0


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
    ) -> Optional[Job]:
        """Submit a new job. Returns None if another job is already in flight.

        Only one job runs at a time — the RVM session, ffmpeg processes,
        and GPU bandwidth are all single-tenant resources, so letting a
        second job overlap just spreads contention without finishing
        either faster.
        """
        with self._lock:
            for j in self.jobs.values():
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    return None
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
        ctx = _RunCtx()
        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._setup(job, ctx)
            canceled = self._loop(job, ctx)
            if canceled:
                job.status = JobStatus.CANCELED
            else:
                self._finalize_normal(ctx)
                job.progress = 1.0
                job.status = JobStatus.DONE
            job.finished_at = time.time()
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.FAILED
            job.finished_at = time.time()
        finally:
            self._cleanup(job, ctx)

    # ---------------- setup ----------------

    def _setup(self, job: Job, ctx: _RunCtx) -> None:
        """Probe input, allocate session/decoder/encoder, start writer thread."""
        info = probe(job.input_path)
        ctx.W, ctx.H, ctx.fps = info["width"], info["height"], info["fps"]
        job.n_frames_total = info["n_frames"]

        ctx.dsr = job.downsample_ratio or suggest_downsample_ratio(ctx.W, ctx.H)

        # Shared session cache. reset() clears any recurrent state left
        # from a previous clip (preview or earlier job).
        ctx.session = get_session(job.model_path)
        ctx.session.reset()
        job.provider = ctx.session.active_provider

        audio_source = None if job.mute else job.input_path
        ctx.decoder = open_decoder(job.input_path)
        if job.output_format == "webm":
            ctx.encoder = open_webm_encoder(
                job.output_path, ctx.W, ctx.H, ctx.fps, audio_source=audio_source
            )
        else:
            ctx.encoder = open_prores_encoder(
                job.output_path, ctx.W, ctx.H, ctx.fps, audio_source=audio_source
            )
        assert ctx.encoder.stdin is not None

        ctx.low, ctx.high, ctx.edge, ctx.feather = clamp_params(
            job.alpha_low, job.alpha_high, job.edge_expand, job.feather
        )
        ctx.do_post = not is_default(ctx.low, ctx.high, ctx.edge, ctx.feather)

        # Background writer thread lets RVM inference overlap with the
        # ffmpeg write+encode for the previous frame. maxsize=2 is
        # enough for steady-state overlap; bigger queues hold more RGBA
        # per slot (33MB at 4K) and stretch cancel latency without
        # adding throughput.
        ctx.encode_q = queue.Queue(maxsize=2)
        ctx.writer_t = threading.Thread(
            target=self._writer, args=(job, ctx), daemon=True
        )
        ctx.writer_t.start()

    @staticmethod
    def _writer(job: Job, ctx: _RunCtx) -> None:
        """Drain encode_q to encoder.stdin; bail on cancel or broken pipe."""
        try:
            while not job._cancel.is_set():
                try:
                    item = ctx.encode_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    return
                try:
                    ctx.encoder.stdin.write(item)
                except (BrokenPipeError, ValueError):
                    return
        except BaseException as exc:  # noqa: BLE001
            ctx.writer_error.append(exc)

    # ---------------- loop ----------------

    def _loop(self, job: Job, ctx: _RunCtx) -> bool:
        """Per-frame infer + enqueue loop. Returns True if canceled mid-stream."""
        t0 = time.time()
        canceled = False
        for idx, rgb in enumerate(iter_frames(ctx.decoder, ctx.W, ctx.H)):
            if job._cancel.is_set():
                canceled = True
                break

            fgr, pha = ctx.session.infer(rgb, ctx.dsr)
            if ctx.do_post:
                pha = postprocess_alpha(pha, ctx.low, ctx.high, ctx.edge, ctx.feather)
            data = np.dstack([fgr, pha]).tobytes()

            # Cancel-polling put: a slow encoder must not stall cancel.
            while True:
                if job._cancel.is_set():
                    canceled = True
                    break
                try:
                    ctx.encode_q.put(data, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if canceled:
                break

            job.n_frames_done = idx + 1
            if job.n_frames_total > 0:
                job.progress = min(1.0, job.n_frames_done / job.n_frames_total)
            dt = time.time() - t0
            if dt > 0:
                job.fps_processed = job.n_frames_done / dt
                remaining = max(0, job.n_frames_total - job.n_frames_done)
                if job.fps_processed > 0:
                    job.eta_seconds = remaining / job.fps_processed
        return canceled

    # ---------------- finalize / cleanup ----------------

    def _finalize_normal(self, ctx: _RunCtx) -> None:
        """Normal-completion drain: flush writer, close stdin, wait subprocesses."""
        ctx.encode_q.put(None)
        ctx.writer_t.join(timeout=60)
        if ctx.writer_error:
            raise ctx.writer_error[0]
        ctx.encoder.stdin.close()
        ctx.encoder.wait(timeout=900)
        ctx.decoder.wait(timeout=60)

    def _cleanup(self, job: Job, ctx: _RunCtx) -> None:
        """Always-run teardown. Each step is independently guarded so a
        failure in one cleanup doesn't skip the next."""
        # Encoder: graceful close already happened on normal flow; on
        # cancel/exception, kill so the writer's pending write fails
        # promptly with BrokenPipeError.
        if ctx.encoder is not None and ctx.encoder.poll() is None:
            try:
                ctx.encoder.kill()
            except Exception:
                pass
            try:
                ctx.encoder.wait(timeout=5)
            except Exception:
                pass
        # Writer thread: encoder is gone, give the writer a moment to
        # notice and exit. The cancel poll also helps it bail.
        if ctx.writer_t is not None and ctx.writer_t.is_alive():
            ctx.writer_t.join(timeout=2)
        # Decoder is still reading the input file; on early exit ffmpeg
        # blocks on its stdout buffer until we kill it.
        if ctx.decoder is not None and ctx.decoder.poll() is None:
            try:
                ctx.decoder.kill()
            except Exception:
                pass
        # Drop a half-written file when the job didn't complete normally.
        # Cancelled jobs may still have wanted the partial output, but
        # the safer default is to discard — users can re-run to keep
        # results.
        if job.status in (JobStatus.CANCELED, JobStatus.FAILED):
            try:
                Path(job.output_path).unlink(missing_ok=True)
            except Exception:
                pass
