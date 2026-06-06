"""Image background removal API (mounted under /api/image)."""
from __future__ import annotations

import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from rembg import remove

from app.image.session import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    get_session,
    loaded_models,
)

router = APIRouter()

# Result cache. WebView2 cannot trigger native downloads via <a download>, so
# the result PNG is held server-side and the client requests it for preview
# and asks the server to write it to disk via /save.
_MAX_RESULTS = 20
_results: "OrderedDict[str, bytes]" = OrderedDict()


def _add_result(data: bytes) -> str:
    result_id = uuid.uuid4().hex[:12]
    _results[result_id] = data
    while len(_results) > _MAX_RESULTS:
        _results.popitem(last=False)
    return result_id


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
) -> dict:
    session = get_session(model)
    raw = await image.read()
    try:
        result = remove(raw, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rembg failed: {exc}") from exc
    return {"result_id": _add_result(result), "model": model}


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
