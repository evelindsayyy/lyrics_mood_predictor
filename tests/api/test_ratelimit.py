"""Rate limiting contract — tiny limit injected via Settings."""

from fastapi.testclient import TestClient

from api.config import Settings
from tests.conftest import FakeMoodModel, FakeRetrieval


def _client(rate_limit="2/minute"):
    from api.main import create_app

    app = create_app(
        settings=Settings(rate_limit=rate_limit),
        models={"baseline": FakeMoodModel()},
        default="baseline",
        retrieval=FakeRetrieval(),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_over_limit_returns_429_envelope():
    c = _client(rate_limit="2/minute")
    for _ in range(2):
        assert c.post("/v1/predict", json={"lyrics": "stadium lights"}).status_code == 200
    r = c.post("/v1/predict", json={"lyrics": "stadium lights"})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    assert "retry-after" in {k.lower() for k in r.headers}


def test_health_is_exempt():
    c = _client(rate_limit="1/minute")
    for _ in range(5):
        assert c.get("/health").status_code == 200


def test_default_limit_generous():
    c = _client(rate_limit="30/minute")
    for _ in range(5):
        assert c.post("/v1/predict", json={"lyrics": "stadium lights"}).status_code == 200
