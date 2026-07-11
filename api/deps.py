"""Dependency accessors — routes depend on app.state, tests inject fakes.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

from fastapi import Request

from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient


def get_models(request: Request) -> dict[str, MoodModel]:
    return request.app.state.models


def get_default_model_name(request: Request) -> str:
    return request.app.state.default_model


def get_retrieval(request: Request) -> RetrievalClient:
    return request.app.state.retrieval


def get_embedder(request: Request):
    return getattr(request.app.state, "embedder", None)


def get_lyrics_store(request: Request):
    return getattr(request.app.state, "lyrics_store", None)
