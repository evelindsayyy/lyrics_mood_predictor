"""Health endpoint tests with injected fakes — no artifacts needed."""

from fastapi.testclient import TestClient

from tests.conftest import FakeMoodModel, FakeRetrieval


def _client(retrieval_ok=True):
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()},
        default="baseline",
        retrieval=FakeRetrieval(ok=retrieval_ok),
    )
    return TestClient(app)


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "qdrant_ok": True,
        "model_version": "fake-v0",
        "models_loaded": ["baseline"],
        "default_model": "baseline",
    }


def test_health_reports_qdrant_down():
    r = _client(retrieval_ok=False).get("/health")
    assert r.status_code == 200
    assert r.json()["qdrant_ok"] is False


def test_request_id_header_echoed():
    client = _client()
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_request_id_generated_when_absent():
    r = _client().get("/health")
    assert len(r.headers["x-request-id"]) >= 8
