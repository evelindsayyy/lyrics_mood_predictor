"""
GET /v1/songs — look a song up by title/artist, run the full mood pipeline
on its lyrics, and return 5 similar songs. Multiple fuzzy matches return
ranked candidates instead. Titles/artists are public metadata and may be
logged; lyrics still never are.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 songs contract, §4 request flow). See ../../ATTRIBUTION.md.
"""

import structlog
from fastapi import APIRouter, Depends, Query

from api.deps import get_default_model_name, get_embedder, get_lyrics_store, get_models, get_retrieval
from api.errors import ApiError
from api.schemas import SongAnalysis, SongResult, SongsResponse
from api.services.embedder import strip_section_headers

router = APIRouter()
logger = structlog.get_logger()


@router.get("/songs", response_model=SongsResponse)
def songs(
    title: str = Query(min_length=2, max_length=200),
    artist: str | None = Query(None, max_length=200),
    retrieval=Depends(get_retrieval),
    embedder=Depends(get_embedder),
    lyrics_store=Depends(get_lyrics_store),
    models=Depends(get_models),
    default_name: str = Depends(get_default_model_name),
) -> SongsResponse:
    try:
        hits = retrieval.find_song(title, artist=artist)
    except Exception as exc:
        logger.warning("retrieval_error", error=type(exc).__name__)
        raise ApiError(503, "retrieval_unavailable", "song lookup is unavailable") from exc

    if not hits:
        raise ApiError(404, "song_not_found", f"no song matching title {title!r}")

    to_result = lambda h: SongResult(**h.__dict__)  # noqa: E731

    if len(hits) > 1:
        logger.info("songs_candidates", title=title, n=len(hits))
        return SongsResponse(match=None, analysis=None, similar=[], candidates=[to_result(h) for h in hits])

    hit = hits[0]
    if lyrics_store is None:
        raise ApiError(503, "lyrics_unavailable", "lyrics store not loaded")
    lyrics = lyrics_store.get(hit.song_id)
    if lyrics is None:
        raise ApiError(404, "song_not_found", f"lyrics missing for song_id {hit.song_id}")

    model = models[default_name]
    result = model.predict(lyrics, explain=False)
    analysis = SongAnalysis(
        mood=result.mood,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_version=model.version,
    )

    similar: list[SongResult] = []
    if embedder is not None:
        vector = embedder.embed([strip_section_headers(lyrics)])[0]
        try:
            raw = retrieval.search(vector, limit=6, mood=result.mood)
            similar = [to_result(h) for h in raw if h.song_id != hit.song_id][:5]
        except Exception as exc:
            logger.warning("retrieval_error", error=type(exc).__name__)

    logger.info("songs_match", title=hit.title, song_id=hit.song_id, mood=result.mood)
    return SongsResponse(match=to_result(hit), analysis=analysis, similar=similar, candidates=[])
