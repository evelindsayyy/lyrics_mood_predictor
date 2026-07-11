"""Prometheus metrics contract."""

from fastapi.testclient import TestClient

from tests.conftest import FakeMoodModel, FakeRetrieval


def _client():
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()}, default="baseline", retrieval=FakeRetrieval()
    )
    return TestClient(app, raise_server_exceptions=False)


def test_metrics_endpoint_exposes_prometheus_text():
    c = _client()
    c.post("/v1/predict", json={"lyrics": "stadium lights"})
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "lyricmood_requests_total" in r.text
    assert 'path="/v1/predict"' in r.text
    assert 'status="200"' in r.text


def test_metrics_uses_route_template_not_raw_path():
    c = _client()
    c.get("/v1/search?q=some+very+specific+query+string")  # 503 (no embedder) — still counted
    r = c.get("/metrics")
    assert "specific+query" not in r.text  # no raw-path/query cardinality


def test_error_statuses_counted():
    c = _client()
    c.post("/v1/predict", json={"lyrics": "   "})  # 400
    r = c.get("/metrics")
    assert 'status="400"' in r.text
