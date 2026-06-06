"""FastAPI app: serves tab UI and mounts image + video routers."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.image.api import router as image_router
from app.video.api import router as video_router

INDEX_PATH = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="bgremove-native")
app.include_router(image_router, prefix="/api/image")
app.include_router(video_router, prefix="/api/video")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")
