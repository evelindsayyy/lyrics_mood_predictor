"""Error envelope contract tests via a minimal throwaway app."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


def _make_app():
    from api.errors import ApiError, register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)

    class Body(BaseModel):
        n: int

    @app.post("/boom-validation")
    def needs_int(body: Body):
        return body

    @app.get("/boom-api-error")
    def api_error():
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")

    @app.get("/boom-crash")
    def crash():
        raise RuntimeError("unexpected")

    return app


def test_api_error_envelope():
    client = TestClient(_make_app())
    r = client.get("/boom-api-error")
    assert r.status_code == 400
    assert r.json() == {
        "error": {"code": "empty_lyrics", "message": "lyrics must contain non-whitespace text"}
    }


def test_validation_error_envelope():
    client = TestClient(_make_app())
    r = client.post("/boom-validation", json={"n": "not-an-int"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert "n" in body["error"]["message"]


def test_unhandled_error_envelope():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    r = client.get("/boom-crash")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "internal server error"}}
