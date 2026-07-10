"""Tests for per-request model selection via ?model=."""

from fastapi.testclient import TestClient

from tests.conftest import FakeMoodModel, FakeRetrieval


def _client():
    from api.main import create_app

    app = create_app(
        models={
            "baseline": FakeMoodModel(mood="Hype", confidence=0.9),
            "transformer": FakeMoodModel(mood="Sad", confidence=0.7),
        },
        default="baseline",
        retrieval=FakeRetrieval(),
    )
    app.state.registry_names = {"baseline", "transformer", "future"}
    return TestClient(app, raise_server_exceptions=False)


def test_default_model_used_without_param():
    r = _client().post("/v1/predict", json={"lyrics": "stadium lights"})
    assert r.status_code == 200
    assert r.json()["mood"] == "Hype"


def test_explicit_model_param_selects_model():
    r = _client().post("/v1/predict?model=transformer", json={"lyrics": "rain empty street"})
    assert r.status_code == 200
    body = r.json()
    assert body["mood"] == "Sad"
    assert body["model_version"] == "fake-v0"


def test_unknown_model_400():
    r = _client().post("/v1/predict?model=nonsense", json={"lyrics": "hello"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_model"


def test_registered_but_unloaded_model_503():
    # "future" is in the registry but its artifacts are not loaded
    r = _client().post("/v1/predict?model=future", json={"lyrics": "hello"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable"


def test_health_reports_loaded_models():
    r = _client().get("/health")
    body = r.json()
    assert body["default_model"] == "baseline"
    assert set(body["models_loaded"]) == {"baseline", "transformer"}
    assert body["model_version"] == "fake-v0"
