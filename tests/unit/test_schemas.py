"""Tests for api.schemas validation boundaries."""

import pytest
from pydantic import ValidationError


def test_predict_request_accepts_lyrics():
    from api.schemas import PredictRequest

    assert PredictRequest(lyrics="stadium lights").lyrics == "stadium lights"


def test_predict_request_rejects_oversize():
    from api.schemas import PredictRequest

    with pytest.raises(ValidationError):
        PredictRequest(lyrics="x" * 10_001)


def test_predict_request_allows_empty_string():
    # empty/whitespace becomes a 400 in the route, not a schema 422
    from api.schemas import PredictRequest

    assert PredictRequest(lyrics="").lyrics == ""


def test_predict_response_shape():
    from api.schemas import PredictResponse, TokenWeight

    r = PredictResponse(
        mood="Hype",
        confidence=0.9,
        probabilities={"Hype": 0.9, "Sad": 0.1},
        explanation=[TokenWeight(token="stadium", weight=0.5)],
        model_version="baseline-lr-v1",
        warnings=[],
    )
    assert r.explanation[0].token == "stadium"


def test_predict_response_explanation_nullable():
    from api.schemas import PredictResponse

    r = PredictResponse(
        mood="Hype",
        confidence=0.9,
        probabilities={"Hype": 0.9},
        explanation=None,
        model_version="v",
        warnings=["input may be non-English"],
    )
    assert r.explanation is None
