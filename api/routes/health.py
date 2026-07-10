"""
GET /health — liveness plus dependency status.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../../ATTRIBUTION.md.
"""

from fastapi import APIRouter, Depends

from api.deps import get_default_model_name, get_models, get_retrieval
from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient

router = APIRouter()


@router.get("/health")
def health(
    models: dict[str, MoodModel] = Depends(get_models),
    default_name: str = Depends(get_default_model_name),
    retrieval: RetrievalClient = Depends(get_retrieval),
):
    return {
        "status": "ok",
        "model_loaded": default_name in models,
        "qdrant_ok": retrieval.ping(),
        "model_version": models[default_name].version,
        "models_loaded": sorted(models),
        "default_model": default_name,
    }
