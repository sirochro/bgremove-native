"""Image background removal API (mounted under /api/image)."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from rembg import remove

from app.alpha import clamp_params, is_default, postprocess_alpha
from app.blob_cache import BlobCache
from app.image.session import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    get_session,
    loaded_models,
)
from app.path_validation import validate_input_path, validate_output_path

router = APIRouter()

# Result blob cache, bounded by both count and total bytes (500 MB).
_results = BlobCache(max_count=20, max_bytes=500 * 1024 * 1024)

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


@router.get("/source")
def get_source(path: str) -> Response:
    """Serve a local image file for preview (by full path)."""
    p = validate_input_path(path, "image")
    mime = _MIME_BY_EXT.get(p.suffix.lower(), "application/octet-stream")
    return Response(content=p.read_bytes(), media_type=mime)


class RemoveRequest(BaseModel):
    input_path: str
    model: str = DEFAULT_MODEL
    alpha_low: int = Field(0, ge=0, le=255)
    alpha_high: int = Field(255, ge=0, le=255)
    edge_expand: int = Field(0, ge=-5, le=5)
    feather: int = Field(0, ge=0, le=20)


@router.post("/remove")
def remove_bg(req: RemoveRequest) -> dict:
    src = validate_input_path(req.input_path, "image")
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

    return {"result_id": _results.add(png_bytes), "model": req.model}


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
    out = validate_output_path(req.output_path, "image")
    out.write_bytes(data)
    return {"ok": True, "bytes": len(data)}
