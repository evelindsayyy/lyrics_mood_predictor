"""
GET /v1/search — free-text mood search; POST /v1/similar — paste-lyrics
similar songs. Both: embed with the ONNX MiniLM -> vector search in Qdrant.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 search contract, §3.3 query mapping). See ../../ATTRIBUTION.md.
"""

import time

import structlog
from fastapi import APIRouter, Depends, Query

from api.deps import get_embedder, get_retrieval
from api.errors import ApiError
from api.schemas import SearchResponse, SimilarRequest, SongResult
from api.services.embedder import strip_section_headers

router = APIRouter()
logger = structlog.get_logger()

MOODS = {"Hype", "Romantic", "Calm", "Sad", "Angry"}


def _validate_mood(mood: str | None) -> None:
    if mood is not None and mood not in MOODS:
        raise ApiError(400, "invalid_mood", f"mood must be one of {sorted(MOODS)}")


def _run_query(text: str, mood: str | None, limit: int, embedder, retrieval) -> SearchResponse:
    if embedder is None:
        raise ApiError(503, "search_unavailable", "query embedder not loaded")
    t0 = time.perf_counter()
    vector = embedder.embed([text])[0]
    t_embed = time.perf_counter()
    try:
        hits = retrieval.search(vector, limit=limit, mood=mood)
    except Exception as exc:
        logger.warning("retrieval_error", error=type(exc).__name__)
        raise ApiError(503, "retrieval_unavailable", "vector search is unavailable") from exc
    t1 = time.perf_counter()
    return SearchResponse(
        results=[SongResult(**h.__dict__) for h in hits],
        query_embedding_ms=(t_embed - t0) * 1000,
        total_ms=(t1 - t0) * 1000,
    )


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=3, max_length=200),
    limit: int = Query(10, ge=1, le=20),
    mood: str | None = Query(None),
    embedder=Depends(get_embedder),
    retrieval=Depends(get_retrieval),
) -> SearchResponse:
    if not q.strip():
        raise ApiError(400, "empty_query", "q must contain non-whitespace text")
    _validate_mood(mood)
    response = _run_query(q, mood, limit, embedder, retrieval)
    logger.info("search", query_chars=len(q), n_results=len(response.results))
    return response


@router.post("/similar", response_model=SearchResponse)
def similar(
    req: SimilarRequest,
    embedder=Depends(get_embedder),
    retrieval=Depends(get_retrieval),
) -> SearchResponse:
    text = strip_section_headers(req.lyrics).strip()
    if not text:
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")
    _validate_mood(req.mood)
    response = _run_query(text, req.mood, req.limit, embedder, retrieval)
    logger.info("similar", input_chars=len(text), n_results=len(response.results))
    return response
