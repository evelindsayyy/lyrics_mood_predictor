"""GET /health — liveness plus dependency status."""

from fastapi import APIRouter, Depends

from api.deps import get_model, get_retrieval
from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient

router = APIRouter()


@router.get("/health")
def health(
    model: MoodModel = Depends(get_model),
    retrieval: RetrievalClient = Depends(get_retrieval),
):
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "qdrant_ok": retrieval.ping(),
        "model_version": model.version,
    }
