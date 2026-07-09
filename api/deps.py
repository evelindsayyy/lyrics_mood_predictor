"""
Dependency accessors — routes depend on app.state, tests inject fakes.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

from fastapi import Request

from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient


def get_model(request: Request) -> MoodModel:
    return request.app.state.model


def get_retrieval(request: Request) -> RetrievalClient:
    return request.app.state.retrieval
