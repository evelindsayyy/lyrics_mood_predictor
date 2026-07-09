"""Contract tests for POST /v1/predict using FakeMoodModel."""

from fastapi.testclient import TestClient

from tests.conftest import FakeMoodModel, FakeRetrieval


def _client(model=None):
    from api.main import create_app

    app = create_app(model=model or FakeMoodModel(), retrieval=FakeRetrieval())
    return TestClient(app, raise_server_exceptions=False)


def test_predict_happy_path():
    r = _client().post("/v1/predict", json={"lyrics": "stadium lights bass kicking"})
    assert r.status_code == 200
    body = r.json()
    assert body["mood"] == "Hype"
    assert body["confidence"] == 0.9
    assert body["probabilities"] == {"Hype": 0.9}
    assert body["explanation"] == [{"token": "stadium", "weight": 0.5}]
    assert body["model_version"] == "fake-v0"
    assert body["warnings"] == []


def test_predict_explain_false():
    r = _client().post("/v1/predict?explain=false", json={"lyrics": "stadium lights"})
    assert r.status_code == 200
    assert r.json()["explanation"] is None


def test_predict_empty_lyrics_400():
    r = _client().post("/v1/predict", json={"lyrics": "   "})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_lyrics"


def test_predict_oversize_422():
    r = _client().post("/v1/predict", json={"lyrics": "x" * 10_001})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_predict_missing_field_422():
    r = _client().post("/v1/predict", json={})
    assert r.status_code == 422


def test_predict_non_latin_warning():
    r = _client().post("/v1/predict", json={"lyrics": "心碎的夜晚 眼泪不停地流 想念你的温柔"})
    assert r.status_code == 200
    assert "input may be non-English" in r.json()["warnings"]


def test_predict_model_failure_500_envelope():
    r = _client(model=FakeMoodModel(fail=True)).post("/v1/predict", json={"lyrics": "hello world"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert r.headers["x-request-id"]
