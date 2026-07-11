"""Contract tests for GET /v1/songs."""

from fastapi.testclient import TestClient

from api.services.retrieval import SongHit
from api.services.songs import LyricsStore
from tests.conftest import FakeEmbedder, FakeMoodModel, FakeRetrieval

ONE = [SongHit(song_id=0, title="Midnight Rain", artist="Ava", mood="Sad", score=0.95)]
MANY = ONE + [SongHit(song_id=1, title="Midnight Train", artist="Bo", mood="Sad", score=0.81)]
SIMILAR = [
    SongHit(song_id=0, title="Midnight Rain", artist="Ava", mood="Sad", score=1.0),  # self
    SongHit(song_id=7, title="Grey Sky", artist="Cy", mood="Sad", score=0.88),
]


def _client(find_hits, store=LyricsStore(["rain empty street lyrics"]), embedder=FakeEmbedder()):
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel(mood="Sad", confidence=0.7)},
        default="baseline",
        retrieval=FakeRetrieval(hits=SIMILAR, find_hits=find_hits),
        embedder=embedder,
        lyrics_store=store,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_single_match_full_analysis():
    r = _client(ONE).get("/v1/songs?title=Midnight Rain")
    assert r.status_code == 200
    body = r.json()
    assert body["match"]["title"] == "Midnight Rain"
    assert body["analysis"]["mood"] == "Sad"
    assert body["analysis"]["model_version"] == "fake-v0"
    # self excluded from similar
    assert [s["song_id"] for s in body["similar"]] == [7]
    assert body["candidates"] == []


def test_multiple_matches_candidates_only():
    r = _client(MANY).get("/v1/songs?title=Midnight")
    body = r.json()
    assert body["match"] is None and body["analysis"] is None
    assert [c["title"] for c in body["candidates"]] == ["Midnight Rain", "Midnight Train"]


def test_no_match_404():
    r = _client([]).get("/v1/songs?title=Nonexistent Song")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "song_not_found"


def test_missing_title_422():
    assert _client(ONE).get("/v1/songs").status_code == 422


def test_lyrics_store_absent_503():
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()}, default="baseline",
        retrieval=FakeRetrieval(find_hits=ONE), embedder=FakeEmbedder(),
    )
    app.state.lyrics_store = None
    r = TestClient(app, raise_server_exceptions=False).get("/v1/songs?title=Midnight Rain")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "lyrics_unavailable"


def test_embedder_absent_still_analyzes():
    r = _client(ONE, embedder=None)
    # embedder=None param means create_app doesn't set it; force-absent:
    r.app.state.embedder = None
    resp = r.get("/v1/songs?title=Midnight Rain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis"]["mood"] == "Sad"
    assert body["similar"] == []


def test_retrieval_down_503():
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()}, default="baseline",
        retrieval=FakeRetrieval(ok=False), embedder=FakeEmbedder(),
        lyrics_store=LyricsStore(["x"]),
    )
    r = TestClient(app, raise_server_exceptions=False).get("/v1/songs?title=Midnight")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "retrieval_unavailable"
