"""
Pydantic request/response models for the LyricMood API.

AI attribution: implementation by Claude (Anthropic) based on my specification
(field names and validation boundaries from the design spec). See ../ATTRIBUTION.md.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    lyrics: str = Field(max_length=10_000)


class TokenWeight(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    mood: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[TokenWeight] | None
    model_version: str
    warnings: list[str]


class SongResult(BaseModel):
    song_id: int
    title: str
    artist: str
    mood: str
    score: float


class SearchResponse(BaseModel):
    results: list[SongResult]
    query_embedding_ms: float
    total_ms: float


class SimilarRequest(BaseModel):
    lyrics: str = Field(max_length=10_000)
    mood: str | None = None
    limit: int = Field(5, ge=1, le=20)
