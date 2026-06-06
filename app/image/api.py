"""Image background removal API (mounted under /api/image)."""
from __future__ import annotations

import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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


@router.post("/remove")
async def remove_bg(
    image: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    alpha_low: int = Form(0),
    alpha_high: int = Form(255),
    edge_expand: int = Form(0),
    feather: int = Form(0),
) -> dict:
    session = get_session(model)
    raw = await image.read()
    try:
        png_bytes = remove(raw, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rembg failed: {exc}") from exc

    low, high, edge, feath = clamp_params(alpha_low, alpha_high, edge_expand, feather)
    if not is_default(low, high, edge, feath):
        png_bytes = _apply_alpha_postprocess(png_bytes, low, high, edge, feath)

    return {"result_id": _add_result(png_bytes), "model": model}


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
