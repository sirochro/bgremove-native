"""Image background removal API (mounted under /api/image)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from rembg import remove

from app.image.session import AVAILABLE_MODELS, DEFAULT_MODEL, get_session, loaded_models

router = APIRouter()


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
) -> Response:
    session = get_session(model)
    raw = await image.read()
    try:
        result = remove(raw, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rembg failed: {exc}") from exc
    return Response(content=result, media_type="image/png")
