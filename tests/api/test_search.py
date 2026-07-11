"""Contract tests for GET /v1/search and POST /v1/similar."""

from fastapi.testclient import TestClient

from api.services.retrieval import SongHit
from tests.conftest import FakeEmbedder, FakeMoodModel, FakeRetrieval

HITS = [
    SongHit(song_id=1, title="Stadium Anthem", artist="The Crowd", mood="Hype", score=0.91),
    SongHit(song_id=2, title="Midnight Rain", artist="Ava", mood="Sad", score=0.80),
]


def _client(retrieval=None, embedder=FakeEmbedder()):
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()},
        default="baseline",
        retrieval=retrieval or FakeRetrieval(hits=HITS),
        embedder=embedder,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_search_happy_path():
    r = _client().get("/v1/search?q=rainy late night drive")
    assert r.status_code == 200
    body = r.json()
    assert [x["title"] for x in body["results"]] == ["Stadium Anthem", "Midnight Rain"]
    assert body["query_embedding_ms"] >= 0
    assert body["total_ms"] >= body["query_embedding_ms"]


def test_search_mood_filter():
    r = _client().get("/v1/search?q=rainy late night drive&mood=Sad")
    assert [x["mood"] for x in r.json()["results"]] == ["Sad"]


def test_search_invalid_mood_400():
    r = _client().get("/v1/search?q=rainy drive&mood=Moody")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_mood"


def test_search_query_too_short_422():
    r = _client().get("/v1/search?q=ab")
    assert r.status_code == 422


def test_search_embedder_absent_503():
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()}, default="baseline", retrieval=FakeRetrieval(hits=HITS)
    )
    app.state.embedder = None
    r = TestClient(app, raise_server_exceptions=False).get("/v1/search?q=rainy drive")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "search_unavailable"


def test_search_retrieval_down_503():
    r = _client(retrieval=FakeRetrieval(ok=False)).get("/v1/search?q=rainy drive")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "retrieval_unavailable"


def test_similar_happy_path():
    r = _client().post("/v1/similar", json={"lyrics": "[Chorus] rain on the street " * 20})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_similar_empty_lyrics_400():
    r = _client().post("/v1/similar", json={"lyrics": "   "})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_lyrics"


def test_similar_mood_filter_and_limit():
    r = _client().post("/v1/similar", json={"lyrics": "rain street", "mood": "Hype", "limit": 1})
    body = r.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["mood"] == "Hype"


def test_search_whitespace_only_q_400():
    r = _client().get("/v1/search?q=%20%20%20%20")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_query"
