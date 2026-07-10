"""
POST /v1/predict — lyrics in, mood + confidence + explanation out.

Sync `def` route: FastAPI runs it in the thread pool, keeping the event
loop free while sklearn/SHAP work.

AI attribution: implementation by Claude (Anthropic) based on my specification
(contract from design spec §3.1/§5, including the non-English warning
heuristic for the known clean_text Latin-script limitation). See ../../ATTRIBUTION.md.
"""

import structlog
from fastapi import APIRouter, Depends, Query, Request

from api.deps import get_default_model_name, get_models
from api.errors import ApiError
from api.schemas import PredictRequest, PredictResponse, TokenWeight
from api.services.model import MoodModel

router = APIRouter()
logger = structlog.get_logger()

NON_LATIN_MAX_ASCII_FRACTION = 0.5


def non_english_warnings(text: str) -> list[str]:
    """Cheap heuristic: mostly non-ASCII letters → probably not English."""
    letters = [c for c in text if c.isalpha()]
    if letters:
        ascii_fraction = sum(c.isascii() for c in letters) / len(letters)
        if ascii_fraction < NON_LATIN_MAX_ASCII_FRACTION:
            return ["input may be non-English"]
    return []


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    explain: bool = Query(True),
    model: str | None = Query(None),
    models: dict[str, MoodModel] = Depends(get_models),
    default_name: str = Depends(get_default_model_name),
    request: Request = None,
) -> PredictResponse:
    text = req.lyrics.strip()
    if not text:
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")

    name = model or default_name
    if name not in models:
        registry_names = getattr(request.app.state, "registry_names", set(models))
        if name in registry_names:
            raise ApiError(503, "model_unavailable", f"model {name!r} is registered but not loaded")
        raise ApiError(400, "unknown_model", f"unknown model {name!r}")

    chosen = models[name]
    result = chosen.predict(text, explain=explain)
    logger.info("predict", input_chars=len(text), mood=result.mood, model=chosen.version)

    return PredictResponse(
        mood=result.mood,
        confidence=result.confidence,
        probabilities=result.probabilities,
        explanation=None
        if result.explanation is None
        else [TokenWeight(token=t, weight=w) for t, w in result.explanation],
        model_version=chosen.version,
        warnings=non_english_warnings(text),
    )
