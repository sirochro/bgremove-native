"""Image background removal API (mounted under /api/image)."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from rembg import remove

from app.alpha import clamp_params, is_default, postprocess_alpha
from app.image.session import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    get_session,
    loaded_models,
)

router = APIRouter()

_MAX_RESULTS = 20
_results: "OrderedDict[str, bytes]" = OrderedDict()

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _add_result(data: bytes) -> str:
    result_id = uuid.uuid4().hex[:12]
    _results[result_id] = data
    while len(_results) > _MAX_RESULTS:
        _results.popitem(last=False)
    return result_id


def _apply_alpha_postprocess(
    png_bytes: bytes, low: int, high: int, edge: int, feather: int
) -> bytes:
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    alpha = postprocess_alpha(alpha, low, high, edge, feather)
    out = np.dstack([rgb, alpha])
    buf = BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


@router.get("/health")
def health() -> dict:
    return {"loaded": loaded_models()}


@router.get("/models")
def list_models() -> dict:
    return {
        "models": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL,
        "loaded": loaded_models(),
    }


@router.post("/upload")
async def upload_image(image: UploadFile = File(...)) -> dict:
    """Save a dropped image to a temp file and return its full path."""
    suffix = Path(image.filename or "image.png").suffix or ".png"
    tmpdir = Path(tempfile.gettempdir()) / "bgremove_uploads"
    tmpdir.mkdir(exist_ok=True)
    dest = tmpdir / f"{uuid.uuid4().hex[:8]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(image.file, fh)
    return {"path": str(dest), "name": image.filename}


@router.get("/source")
def get_source(path: str) -> Response:
    """Serve a local image file for preview (by full path)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Source not found")
    mime = _MIME_BY_EXT.get(p.suffix.lower(), "application/octet-stream")
    return Response(content=p.read_bytes(), media_type=mime)


class RemoveRequest(BaseModel):
    input_path: str
    model: str = DEFAULT_MODEL
    alpha_low: int = 0
    alpha_high: int = 255
    edge_expand: int = 0
    feather: int = 0


@router.post("/remove")
def remove_bg(req: RemoveRequest) -> dict:
    src = Path(req.input_path)
    if not src.exists():
        raise HTTPException(400, f"Input not found: {req.input_path}")
    session = get_session(req.model)
    raw = src.read_bytes()
    try:
        png_bytes = remove(raw, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rembg failed: {exc}") from exc

    low, high, edge, feath = clamp_params(
        req.alpha_low, req.alpha_high, req.edge_expand, req.feather
    )
    if not is_default(low, high, edge, feath):
        png_bytes = _apply_alpha_postprocess(png_bytes, low, high, edge, feath)

    return {"result_id": _add_result(png_bytes), "model": req.model}


@router.get("/result/{result_id}")
def get_result(result_id: str) -> Response:
    data = _results.get(result_id)
    if data is None:
        raise HTTPException(404, "Result not found")
    return Response(content=data, media_type="image/png")


class SaveRequest(BaseModel):
    result_id: str
    output_path: str


@router.post("/save")
def save_result(req: SaveRequest) -> dict:
    data = _results.get(req.result_id)
    if data is None:
        raise HTTPException(404, "Result not found")
    out = Path(req.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {"ok": True, "bytes": len(data)}
