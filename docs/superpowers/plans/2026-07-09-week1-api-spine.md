# LyricMood Week 1 — API Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dockerized FastAPI service serving `POST /v1/predict` on the existing TF-IDF+LR baseline model, with the error contract, health endpoint, structured logging, tests, and an idempotent Qdrant corpus indexer.

**Architecture:** FastAPI app factory with lifespan-loaded artifacts and dependency injection (`app.state`), so tests swap in fakes without heavy artifacts. Model logic lives in `api/services/` wrapping the existing `src/` modules unchanged. Qdrant runs as a Docker Compose sibling; `scripts/index_corpus.py` populates it from the processed corpus. Routes are sync `def` functions so FastAPI's built-in thread pool handles blocking inference off the event loop.

**Tech Stack:** FastAPI, Pydantic v2 + pydantic-settings, uvicorn, qdrant-client, structlog, joblib/sklearn/shap (existing), pytest + httpx TestClient.

**Spec:** `docs/superpowers/specs/2026-07-09-industrial-elevation-design.md` (this plan implements the "Week 1 — spine" row of §7).

## Global Constraints

- Python 3.10+; free-tier only (no paid services).
- `random_state=42` / `np.random.default_rng(42)` for any sampling.
- Every new module docstring carries an AI-attribution block (per `ATTRIBUTION.md`).
- Error envelope is always `{"error": {"code": "<snake_case>", "message": "<human readable>"}}`.
- Raw lyrics are NEVER logged — log lengths and request IDs only.
- Public endpoints live under `/v1` except `/health`.
- Existing `src/` modules are consumed, not modified, in this plan.
- Rate limiting and `/metrics` are Week 3 — do NOT add them now (YAGNI).

## File Structure

```
api/
├── __init__.py            # empty package marker
├── config.py              # Settings (pydantic-settings, env prefix LYRICMOOD_)
├── errors.py              # ApiError + exception handlers → error envelope
├── schemas.py             # PredictRequest/PredictResponse/TokenWeight
├── deps.py                # get_model / get_retrieval from app.state
├── logging_setup.py       # structlog JSON config + request-id middleware
├── main.py                # create_app() factory + module-level app
├── routes/
│   ├── __init__.py
│   ├── health.py          # GET /health
│   └── predict.py         # POST /v1/predict
└── services/
    ├── __init__.py
    ├── model.py           # MoodModel protocol, PredictionResult, BaselineMoodModel, load_baseline
    └── retrieval.py       # RetrievalClient protocol, QdrantRetrieval (ping only this week)
scripts/index_corpus.py    # idempotent Qdrant ingest (testable core + CLI main)
docker/Dockerfile.api
docker-compose.yml
.dockerignore
requirements-api.txt       # serving runtime deps
requirements-dev.txt       # pytest, httpx (includes -r requirements-api.txt)
pytest.ini
tests/
├── conftest.py            # tiny trained model fixture, FakeMoodModel, FakeRetrieval, app fixture
├── unit/test_config.py
├── unit/test_schemas.py
├── unit/test_model_service.py
├── unit/test_index_corpus.py
├── api/test_errors.py
├── api/test_health.py
└── api/test_predict.py
```

---

### Task 1: Scaffolding, dependencies, Settings

**Files:**
- Create: `requirements-api.txt`, `requirements-dev.txt`, `pytest.ini`, `api/__init__.py`, `api/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (pydantic-settings class) with fields `model_dir: Path`, `baseline_classifier: str`, `baseline_vectorizer: str`, `labeled_songs_path: Path`, `qdrant_url: str`, `qdrant_collection: str`, `shap_background_size: int`, `max_lyrics_chars: int`. Env override prefix `LYRICMOOD_`.

- [ ] **Step 1: Create dependency and pytest config files**

`requirements-api.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
pydantic-settings>=2.2
qdrant-client>=1.9
structlog>=24.1
joblib>=1.2
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
shap>=0.44
```

`requirements-dev.txt`:

```
-r requirements-api.txt
pytest>=8.0
httpx>=0.27
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -q
```

`api/__init__.py`: empty file.

Also create empty package markers so `from tests.conftest import FakeMoodModel` works in later tasks: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/api/__init__.py`.

Run: `source .venv/bin/activate && pip install -r requirements-dev.txt`
Expected: installs cleanly (fastapi, qdrant-client, structlog, pytest, httpx are the new ones).

- [ ] **Step 2: Write the failing test**

`tests/unit/test_config.py`:

```python
"""Tests for api.config.Settings."""


def test_settings_defaults():
    from api.config import Settings

    s = Settings()
    assert str(s.model_dir) == "models"
    assert s.baseline_classifier == "best_classifier.pkl"
    assert s.baseline_vectorizer == "tfidf_vectorizer.pkl"
    assert str(s.labeled_songs_path) == "data/processed/songs_labeled.csv"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "songs"
    assert s.shap_background_size == 500
    assert s.max_lyrics_chars == 10_000


def test_settings_env_override(monkeypatch):
    from api.config import Settings

    monkeypatch.setenv("LYRICMOOD_QDRANT_URL", "http://qdrant:6333")
    assert Settings().qdrant_url == "http://qdrant:6333"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.config'`

- [ ] **Step 4: Write minimal implementation**

`api/config.py`:

```python
"""
Runtime settings for the LyricMood API.

All values overridable via environment variables prefixed LYRICMOOD_
(e.g. LYRICMOOD_QDRANT_URL) so docker-compose can rewire service URLs.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose the setting names, defaults, and the env-prefix convention; Claude
wrote the class. See ../ATTRIBUTION.md.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LYRICMOOD_")

    model_dir: Path = Path("models")
    baseline_classifier: str = "best_classifier.pkl"
    baseline_vectorizer: str = "tfidf_vectorizer.pkl"
    labeled_songs_path: Path = Path("data/processed/songs_labeled.csv")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "songs"
    shap_background_size: int = 500
    max_lyrics_chars: int = 10_000
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add requirements-api.txt requirements-dev.txt pytest.ini api/__init__.py api/config.py tests/__init__.py tests/unit/__init__.py tests/api/__init__.py tests/unit/test_config.py
git commit -m "feat: add api package scaffolding and settings"
```

---

### Task 2: Request/response schemas

**Files:**
- Create: `api/schemas.py`
- Test: `tests/unit/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PredictRequest(lyrics: str)` (max_length=10_000; no min — empty/whitespace is a 400 at the route, not a 422); `TokenWeight(token: str, weight: float)`; `PredictResponse(mood: str, confidence: float, probabilities: dict[str, float], explanation: list[TokenWeight] | None, model_version: str, warnings: list[str])`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_schemas.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.schemas'`

- [ ] **Step 3: Write minimal implementation**

`api/schemas.py`:

```python
"""
Pydantic request/response models for the LyricMood API.

AI attribution: implementation by Claude (Anthropic) based on my specification
(field names and validation boundaries from the design spec). See ../ATTRIBUTION.md.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    lyrics: str = Field(max_length=10_000)


class TokenWeight(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    mood: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[TokenWeight] | None
    model_version: str
    warnings: list[str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py tests/unit/test_schemas.py
git commit -m "feat: add predict request/response schemas"
```

---

### Task 3: Baseline model service

**Files:**
- Create: `api/services/__init__.py`, `api/services/model.py`
- Create: `tests/conftest.py`
- Test: `tests/unit/test_model_service.py`

**Interfaces:**
- Consumes: `src.preprocess.clean_text(text: str) -> str`, `src.explain.explain_prediction(model, vectorizer, text, top_k, background) -> dict` (existing, unchanged).
- Produces:
  - `PredictionResult` frozen dataclass: `mood: str`, `confidence: float`, `probabilities: dict[str, float]`, `explanation: list[tuple[str, float]] | None`.
  - `MoodModel` Protocol: attribute `version: str`; method `predict(lyrics: str, explain: bool = True) -> PredictionResult`.
  - `BaselineMoodModel(clf, vectorizer, background=None, version="baseline-lr-v1")` implementing `MoodModel`.
  - `ArtifactError(Exception)`.
  - `load_baseline(settings: Settings) -> BaselineMoodModel` — fails fast with `ArtifactError` naming the missing path.
  - Test fixtures in `tests/conftest.py`: `tiny_model` (real sklearn artifacts trained on 15 tiny songs), `FakeMoodModel` class.

- [ ] **Step 1: Write conftest with the tiny-model fixture and FakeMoodModel**

`tests/conftest.py`:

```python
"""
Shared fixtures: a real (tiny) sklearn model for service tests, and a
FakeMoodModel + FakeRetrieval for API route tests so no heavy artifacts
are needed in CI.
"""

import pytest

# 3 songs per mood, words chosen to survive clean_text stopword stripping
TINY_SONGS = [
    ("stadium lights bass kicking loud crowd jumping", "Hype"),
    ("party anthem hands raised speakers booming dance", "Hype"),
    ("energy rising drums pounding neon strobing night", "Hype"),
    ("tender heart kitchen door soft radio lifetime", "Romantic"),
    ("gentle kisses warm embrace candle dinner roses", "Romantic"),
    ("holding hands moonlight promise sweet whisper darling", "Romantic"),
    ("slow light rug cool tea window tree quiet", "Calm"),
    ("morning stillness breeze garden peaceful drifting cloud", "Calm"),
    ("lazy sunday blanket humming kettle soft rain", "Calm"),
    ("rain empty street counted cars coat chair alone", "Sad"),
    ("tears falling goodbye letter fading photograph missing", "Sad"),
    ("grey sky lonely echo hollow rooms winter grief", "Sad"),
    ("face blame burned door shouting fists slammed", "Angry"),
    ("rage boiling betrayal lies screaming broken glass", "Angry"),
    ("fury storm smashed walls venom spite revenge", "Angry"),
]


@pytest.fixture(scope="session")
def tiny_model():
    """A real BaselineMoodModel trained on TINY_SONGS — fast, deterministic."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from api.services.model import BaselineMoodModel

    texts = [t for t, _ in TINY_SONGS]
    labels = [m for _, m in TINY_SONGS]
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, labels)
    return BaselineMoodModel(clf=clf, vectorizer=vec, background=None, version="test-v0")


class FakeMoodModel:
    """Canned-response model for route tests."""

    version = "fake-v0"

    def __init__(self, mood="Hype", confidence=0.9, explanation=(("stadium", 0.5),), fail=False):
        self._mood = mood
        self._confidence = confidence
        self._explanation = list(explanation)
        self._fail = fail

    def predict(self, lyrics, explain=True):
        from api.services.model import PredictionResult

        if self._fail:
            raise RuntimeError("model exploded")
        return PredictionResult(
            mood=self._mood,
            confidence=self._confidence,
            probabilities={self._mood: self._confidence},
            explanation=self._explanation if explain else None,
        )


class FakeRetrieval:
    """Canned retrieval client for route tests."""

    def __init__(self, ok=True):
        self._ok = ok

    def ping(self):
        return self._ok
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_model_service.py`:

```python
"""Tests for api.services.model."""

import pytest


def test_predict_returns_known_mood(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd")
    assert r.mood in {"Hype", "Romantic", "Calm", "Sad", "Angry"}
    assert 0.0 < r.confidence <= 1.0
    assert pytest.approx(sum(r.probabilities.values()), abs=1e-6) == 1.0


def test_predict_hype_lyrics_lean_hype(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd jumping party")
    assert r.mood == "Hype"


def test_explanation_contains_input_tokens(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd", explain=True)
    assert r.explanation is not None
    assert len(r.explanation) <= 10
    tokens = {t for t, _ in r.explanation}
    assert tokens & {"stadium", "lights", "bass", "kicking", "loud", "crowd"}


def test_explain_false_skips_explanation(tiny_model):
    r = tiny_model.predict("stadium lights bass", explain=False)
    assert r.explanation is None


def test_no_vocab_overlap_explanation_is_none_or_empty(tiny_model):
    # words absent from tiny vocabulary → nothing for SHAP to rank
    r = tiny_model.predict("zzzz qqqq xxxx", explain=True)
    assert r.explanation in (None, [])


def test_load_baseline_missing_artifact_fails_fast(tmp_path):
    from api.config import Settings
    from api.services.model import ArtifactError, load_baseline

    s = Settings(model_dir=tmp_path)  # empty dir — no pickles
    with pytest.raises(ArtifactError) as exc:
        load_baseline(s)
    assert "best_classifier.pkl" in str(exc.value)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_model_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services'`

- [ ] **Step 4: Write the implementation**

`api/services/__init__.py`: empty file.

`api/services/model.py`:

```python
"""
Model service layer — wraps the existing src/ baseline (TF-IDF + LR + SHAP)
behind a MoodModel protocol so routes and tests depend on an interface,
not on artifacts. The fine-tuned transformer (Week 2) will implement the
same protocol.

AI attribution: implementation by Claude (Anthropic) based on my specification
(protocol shape, fail-fast artifact loading, explanation semantics carried
over from app/streamlit_app.py). See ../ATTRIBUTION.md.
"""

from dataclasses import dataclass
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from api.config import Settings
from src.explain import explain_prediction
from src.preprocess import clean_text


class ArtifactError(Exception):
    """A required model artifact is missing or unreadable."""


@dataclass(frozen=True)
class PredictionResult:
    mood: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[tuple[str, float]] | None


class MoodModel(Protocol):
    version: str

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult: ...


class BaselineMoodModel:
    """TF-IDF + logistic regression with exact SHAP explanations."""

    def __init__(self, clf, vectorizer, background=None, version: str = "baseline-lr-v1"):
        self._clf = clf
        self._vectorizer = vectorizer
        self._background = background
        self.version = version

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult:
        cleaned = clean_text(lyrics)
        X = self._vectorizer.transform([cleaned])
        probs = self._clf.predict_proba(X)[0]
        classes = list(self._clf.classes_)
        idx = int(np.argmax(probs))
        explanation = self._explain(cleaned, X) if explain else None
        return PredictionResult(
            mood=str(classes[idx]),
            confidence=float(probs[idx]),
            probabilities={str(c): float(p) for c, p in zip(classes, probs)},
            explanation=explanation,
        )

    def _explain(self, cleaned: str, X) -> list[tuple[str, float]] | None:
        """Top-10 input tokens by |SHAP|; None on any failure (non-fatal per spec)."""
        try:
            exp = explain_prediction(
                self._clf, self._vectorizer, cleaned, top_k=10, background=self._background
            )
            sv = exp["shap_values"]
            fn = exp["feature_names"]
            present = X.nonzero()[1]
            pairs = [(str(fn[i]), float(sv[i])) for i in present]
            top = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:10]
            return sorted(top, key=lambda kv: kv[1], reverse=True)
        except Exception:
            return None


def load_baseline(settings: Settings) -> BaselineMoodModel:
    """Load pickled artifacts; fail fast with the offending path in the message."""
    clf_path = settings.model_dir / settings.baseline_classifier
    vec_path = settings.model_dir / settings.baseline_vectorizer
    for path in (clf_path, vec_path):
        if not path.exists():
            raise ArtifactError(f"model artifact missing: {path}")
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    background = None
    if settings.labeled_songs_path.exists():
        df = pd.read_csv(settings.labeled_songs_path, usecols=["lyrics"])
        rng = np.random.default_rng(42)
        n = min(settings.shap_background_size, len(df))
        bg_idx = rng.choice(len(df), size=n, replace=False)
        background = vec.transform(df["lyrics"].iloc[bg_idx].map(clean_text))

    return BaselineMoodModel(clf=clf, vectorizer=vec, background=background)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_model_service.py -v`
Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add api/services/__init__.py api/services/model.py tests/conftest.py tests/unit/test_model_service.py
git commit -m "feat: add baseline model service behind MoodModel protocol"
```

---

### Task 4: Error envelope and exception handlers

**Files:**
- Create: `api/errors.py`
- Test: `tests/api/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ApiError(status_code: int, code: str, message: str)` exception; `register_exception_handlers(app: FastAPI) -> None` installing handlers for `ApiError` (its status), `RequestValidationError` (422), and bare `Exception` (500, code `internal_error`). All responses use the envelope `{"error": {"code", "message"}}`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_errors.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.errors'`

- [ ] **Step 3: Write minimal implementation**

`api/errors.py`:

```python
"""
Error contract: every error response is {"error": {"code", "message"}}.

AI attribution: implementation by Claude (Anthropic) based on my specification
(envelope shape and code taxonomy from the design spec §5). See ../ATTRIBUTION.md.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        logger.warning("api_error", code=exc.code, status=exc.status_code)
        return _envelope(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        message = f"{loc}: {first.get('msg', 'invalid input')}"
        logger.warning("validation_error", detail=message)
        return _envelope(422, "validation_error", message)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.error("internal_error", exc_info=exc)
        return _envelope(500, "internal_error", "internal server error")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_errors.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add api/errors.py tests/api/test_errors.py
git commit -m "feat: add error envelope and exception handlers"
```

---

### Task 5: Retrieval ping, logging/request-id, app factory, /health

**Files:**
- Create: `api/services/retrieval.py`, `api/logging_setup.py`, `api/deps.py`, `api/routes/__init__.py`, `api/routes/health.py`, `api/main.py`
- Test: `tests/api/test_health.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `MoodModel`/`load_baseline`/`FakeMoodModel` (Task 3), `register_exception_handlers` (Task 4), `FakeRetrieval` (conftest, Task 3).
- Produces:
  - `RetrievalClient` Protocol: `ping() -> bool`.
  - `QdrantRetrieval(url: str)` implementing it (`ping` returns False on any connection error).
  - `configure_logging() -> None` (structlog JSON to stdout) and `request_id_middleware` (binds `request_id` contextvar, echoes `x-request-id` response header).
  - `get_model(request) -> MoodModel` and `get_retrieval(request) -> RetrievalClient` in `api/deps.py`, reading `request.app.state`.
  - `create_app(settings=None, model=None, retrieval=None) -> FastAPI`; lifespan loads real artifacts only for the ones not injected. Module-level `app = create_app()` in `api/main.py` for uvicorn.
  - `GET /health` → `{"status": "ok", "model_loaded": bool, "qdrant_ok": bool, "model_version": str}`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_health.py`:

```python
"""Health endpoint tests with injected fakes — no artifacts needed."""

from fastapi.testclient import TestClient

from tests.conftest import FakeMoodModel, FakeRetrieval


def _client(retrieval_ok=True):
    from api.main import create_app

    app = create_app(model=FakeMoodModel(), retrieval=FakeRetrieval(ok=retrieval_ok))
    return TestClient(app)


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "qdrant_ok": True,
        "model_version": "fake-v0",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 3: Write the implementation (five small modules)**

`api/services/retrieval.py`:

```python
"""
Retrieval client layer. Week 1 only needs ping() for /health; vector search
and payload lookup land in Week 3 behind this same protocol.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

from typing import Protocol

from qdrant_client import QdrantClient


class RetrievalClient(Protocol):
    def ping(self) -> bool: ...


class QdrantRetrieval:
    def __init__(self, url: str):
        self._client = QdrantClient(url=url, timeout=2)

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
```

`api/logging_setup.py`:

```python
"""
Structured JSON logging + request-id middleware. Raw lyrics are never logged;
handlers log lengths and codes only.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

import logging
import uuid

import structlog
from fastapi import Request


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["x-request-id"] = rid
    return response
```

`api/deps.py`:

```python
"""Dependency accessors — routes depend on app.state, tests inject fakes."""

from fastapi import Request

from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient


def get_model(request: Request) -> MoodModel:
    return request.app.state.model


def get_retrieval(request: Request) -> RetrievalClient:
    return request.app.state.retrieval
```

`api/routes/__init__.py`: empty file.

`api/routes/health.py`:

```python
"""GET /health — liveness plus dependency status."""

from fastapi import APIRouter, Depends

from api.deps import get_model, get_retrieval
from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient

router = APIRouter()


@router.get("/health")
def health(
    model: MoodModel = Depends(get_model),
    retrieval: RetrievalClient = Depends(get_retrieval),
):
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "qdrant_ok": retrieval.ping(),
        "model_version": model.version,
    }
```

`api/main.py`:

```python
"""
App factory. Lifespan loads real artifacts unless fakes are injected
(tests pass model=/retrieval=). Routes are sync `def` so FastAPI runs
them in its thread pool — blocking sklearn/SHAP inference never blocks
the event loop.

AI attribution: implementation by Claude (Anthropic) based on my specification
(factory + state-injection pattern chosen for testability). See ../ATTRIBUTION.md.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import Settings
from api.errors import register_exception_handlers
from api.logging_setup import configure_logging, request_id_middleware
from api.routes import health, predict
from api.services.model import MoodModel, load_baseline
from api.services.retrieval import QdrantRetrieval, RetrievalClient


def create_app(
    settings: Settings | None = None,
    model: MoodModel | None = None,
    retrieval: RetrievalClient | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.model = model if model is not None else load_baseline(cfg)
        app.state.retrieval = retrieval if retrieval is not None else QdrantRetrieval(cfg.qdrant_url)
        yield

    app = FastAPI(title="LyricMood API", version="1.0", lifespan=lifespan)
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    return app


app = create_app()
```

NOTE: `api/main.py` imports `api.routes.predict`, which doesn't exist until Task 6. To keep this task green on its own, create `api/routes/predict.py` now as a stub with just an empty router:

```python
"""POST /v1/predict — implemented in the next task."""

from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_health.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: all tests pass (config, schemas, model service, errors, health).

- [ ] **Step 6: Commit**

```bash
git add api/services/retrieval.py api/logging_setup.py api/deps.py api/routes/__init__.py api/routes/health.py api/routes/predict.py api/main.py tests/api/test_health.py
git commit -m "feat: add app factory, health endpoint, retrieval ping, structured logging"
```

---

### Task 6: POST /v1/predict

**Files:**
- Modify: `api/routes/predict.py` (replace the stub)
- Test: `tests/api/test_predict.py`

**Interfaces:**
- Consumes: `PredictRequest`/`PredictResponse`/`TokenWeight` (Task 2), `MoodModel.predict(lyrics, explain)` → `PredictionResult` (Task 3), `ApiError` (Task 4), `get_model` (Task 5), `FakeMoodModel` (conftest).
- Produces: `POST /v1/predict?explain=true|false` returning `PredictResponse`; helper `non_english_warnings(text: str) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/api/test_predict.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/api/test_predict.py -v`
Expected: FAIL — happy-path and others return 404/405 (stub router has no POST route yet).

- [ ] **Step 3: Replace the stub with the implementation**

`api/routes/predict.py`:

```python
"""
POST /v1/predict — lyrics in, mood + confidence + explanation out.

Sync `def` route: FastAPI runs it in the thread pool, keeping the event
loop free while sklearn/SHAP work.

AI attribution: implementation by Claude (Anthropic) based on my specification
(contract from design spec §3.1/§5, including the non-English warning
heuristic for the known clean_text Latin-script limitation). See ../ATTRIBUTION.md.
"""

import structlog
from fastapi import APIRouter, Depends, Query

from api.deps import get_model
from api.errors import ApiError
from api.schemas import PredictRequest, PredictResponse, TokenWeight
from api.services.model import MoodModel

router = APIRouter()
logger = structlog.get_logger()

NON_LATIN_MAX_ASCII_FRACTION = 0.5


def non_english_warnings(text: str) -> list[str]:
    """Cheap heuristic: mostly non-ASCII letters → probably not English."""
    letters = [c for c in text if c.isalpha()]
    if letters:
        ascii_fraction = sum(c.isascii() for c in letters) / len(letters)
        if ascii_fraction < NON_LATIN_MAX_ASCII_FRACTION:
            return ["input may be non-English"]
    return []


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    explain: bool = Query(True),
    model: MoodModel = Depends(get_model),
) -> PredictResponse:
    text = req.lyrics.strip()
    if not text:
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")

    result = model.predict(text, explain=explain)
    logger.info("predict", input_chars=len(text), mood=result.mood, model=model.version)

    return PredictResponse(
        mood=result.mood,
        confidence=result.confidence,
        probabilities=result.probabilities,
        explanation=None
        if result.explanation is None
        else [TokenWeight(token=t, weight=w) for t, w in result.explanation],
        model_version=model.version,
        warnings=non_english_warnings(text),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_predict.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Run the whole suite and commit**

Run: `pytest`
Expected: all green.

```bash
git add api/routes/predict.py tests/api/test_predict.py
git commit -m "feat: add POST /v1/predict endpoint"
```

---

### Task 7: Idempotent corpus indexer

**Files:**
- Create: `scripts/__init__.py` (empty), `scripts/index_corpus.py`
- Test: `tests/unit/test_index_corpus.py`

**Interfaces:**
- Consumes: `qdrant_client.QdrantClient` (`:memory:` mode in tests), `src.recommend.load_embedding_model`/`embed_corpus` (existing, CLI path only).
- Produces:
  - `strip_section_headers(text) -> str`
  - `resolve_first_artist(ids_str: str, artist_map: dict) -> str`
  - `ensure_collection(client, name) -> None` (384-d cosine vectors; payload indexes: `mood` keyword, `title` text, `artist` text)
  - `index_corpus(client, df, embeddings, collection, batch_size=256) -> int` — deterministic integer point IDs (row position), so re-running upserts in place. `df` must have columns `name`, `artist`, `mood`, `valence`, `energy`, `lyrics`.
  - CLI `python scripts/index_corpus.py` wiring real CSVs + cached embeddings to a real Qdrant.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_index_corpus.py`:

```python
"""Indexer tests against an in-memory Qdrant."""

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def tiny_corpus():
    df = pd.DataFrame(
        {
            "name": ["Song A", "Song B", "Song C"],
            "artist": ["Artist 1", "Artist 2", "Artist 3"],
            "mood": ["Hype", "Sad", "Hype"],
            "valence": [0.9, 0.1, 0.8],
            "energy": [0.9, 0.2, 0.7],
            "lyrics": ["[Chorus] loud crowd jumping", "rain empty street", "bass kicking night"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(3, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return df, emb


def test_strip_section_headers():
    from scripts.index_corpus import strip_section_headers

    assert strip_section_headers("[Chorus] loud crowd") == " loud crowd"
    assert strip_section_headers(None) == ""


def test_resolve_first_artist():
    from scripts.index_corpus import resolve_first_artist

    amap = {"id1": "Taylor", "id2": "Kendrick"}
    assert resolve_first_artist("['id1', 'id2']", amap) == "Taylor"
    assert resolve_first_artist("[]", amap) == "Unknown"
    assert resolve_first_artist("not-a-list", amap) == "Unknown"


def test_index_corpus_upserts_all_rows(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    assert index_corpus(client, df, emb, "songs") == 3
    assert client.count("songs").count == 3


def test_index_corpus_is_idempotent(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    index_corpus(client, df, emb, "songs")  # run twice
    assert client.count("songs").count == 3  # no duplicates


def test_payload_contents_and_excerpt(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    point = client.retrieve("songs", ids=[0], with_payload=True)[0]
    assert point.payload["title"] == "Song A"
    assert point.payload["mood"] == "Hype"
    assert "[Chorus]" not in point.payload["lyrics_excerpt"]
    assert "lyrics" not in point.payload  # full lyrics never enter Qdrant


def test_row_embedding_mismatch_raises(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    with pytest.raises(ValueError, match="row count"):
        index_corpus(client, df, emb[:2], "songs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_index_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.index_corpus'`

- [ ] **Step 3: Write the implementation**

`scripts/__init__.py`: empty file.

`scripts/index_corpus.py`:

```python
"""
Populate the Qdrant `songs` collection from the processed corpus.

Idempotent: point IDs are row positions, so re-running upserts in place.
Full lyrics never enter Qdrant (copyright + size) — only a ~300-char excerpt.

Usage (Qdrant running via docker compose):
    python scripts/index_corpus.py

AI attribution: implementation by Claude (Anthropic) based on my specification
(schema from design spec §3.3; artist resolution logic carried over from
app/streamlit_app.py). See ../ATTRIBUTION.md.
"""

import ast
import re
import sys

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

VECTOR_SIZE = 384
EXCERPT_CHARS = 300


def strip_section_headers(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[[^\]]*\]", " ", text)


def resolve_first_artist(ids_str, artist_map: dict) -> str:
    try:
        ids = ast.literal_eval(ids_str)
        return artist_map.get(ids[0], "Unknown") if ids else "Unknown"
    except (ValueError, SyntaxError):
        return "Unknown"


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        name, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    client.create_payload_index(name, "mood", PayloadSchemaType.KEYWORD)
    text_params = TextIndexParams(type="text", tokenizer=TokenizerType.WORD, lowercase=True)
    client.create_payload_index(name, "title", text_params)
    client.create_payload_index(name, "artist", text_params)


def index_corpus(
    client: QdrantClient,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    collection: str,
    batch_size: int = 256,
) -> int:
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"row count mismatch: {len(df)} corpus rows vs {embeddings.shape[0]} embeddings"
        )
    total = 0
    points: list[PointStruct] = []
    for pos, row in enumerate(df.itertuples(index=False)):
        excerpt = strip_section_headers(row.lyrics).strip()[:EXCERPT_CHARS]
        points.append(
            PointStruct(
                id=pos,
                vector=embeddings[pos].tolist(),
                payload={
                    "song_id": pos,
                    "title": row.name,
                    "artist": row.artist,
                    "mood": row.mood,
                    "valence": float(row.valence),
                    "energy": float(row.energy),
                    "lyrics_excerpt": excerpt,
                },
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection, points=points)
            total += len(points)
            points = []
    if points:
        client.upsert(collection, points=points)
        total += len(points)
    return total


def main() -> int:
    from api.config import Settings
    from src.recommend import embed_corpus, load_embedding_model

    settings = Settings()
    if not settings.labeled_songs_path.exists():
        print(f"missing {settings.labeled_songs_path} — run notebooks/01_eda.ipynb first")
        return 1

    df = pd.read_csv(settings.labeled_songs_path).reset_index(drop=True)
    artists = pd.read_csv("SpotGenTrack/Data Sources/spotify_artists.csv", usecols=["id", "name"])
    artist_map = dict(zip(artists["id"], artists["name"]))
    df["artist"] = df["artists_id"].map(lambda s: resolve_first_artist(s, artist_map))

    raw = df["lyrics"].map(strip_section_headers)
    embeddings = embed_corpus(load_embedding_model(), raw.tolist())  # reuses .npy cache
    if embeddings.shape[0] != len(df):
        print("cached embeddings don't match corpus — delete models/corpus_embeddings.npy and rerun")
        return 1

    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, settings.qdrant_collection)
    n = index_corpus(client, df, embeddings, settings.qdrant_collection)
    print(f"indexed {n} songs into '{settings.qdrant_collection}' at {settings.qdrant_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

NOTE: `df.itertuples(index=False)` + a column named `name` — pandas renames `name` in itertuples only when `index=True`; with `index=False` and unique valid identifiers it is preserved. The test `test_payload_contents_and_excerpt` catches any surprise here; if `row.name` misbehaves on the implementer's pandas version, switch the loop to `df.iterrows()` and `row["name"]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_index_corpus.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/index_corpus.py tests/unit/test_index_corpus.py
git commit -m "feat: add idempotent qdrant corpus indexer"
```

---

### Task 8: Docker packaging

**Files:**
- Create: `docker/Dockerfile.api`, `docker-compose.yml`, `.dockerignore`

**Interfaces:**
- Consumes: `api.main:app` (Task 5), `requirements-api.txt` (Task 1), env override `LYRICMOOD_QDRANT_URL` (Task 1).
- Produces: `docker compose up` → Qdrant on :6333 + API on :8000 with `models/` and `data/processed/` mounted read-only.

- [ ] **Step 1: Write the three files**

`.dockerignore`:

```
.venv
.git
SpotGenTrack
data
models
results
videos
notebooks
docs
tests
__pycache__
*.ipynb
.DS_Store
```

`docker/Dockerfile.api`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.5
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      LYRICMOOD_QDRANT_URL: http://qdrant:6333
    volumes:
      - ./models:/app/models:ro
      - ./data/processed:/app/data/processed:ro
    depends_on:
      - qdrant

volumes:
  qdrant_storage:
```

- [ ] **Step 2: Build and smoke-test locally (requires local artifacts)**

Run: `docker compose build`
Expected: image builds cleanly.

Run: `docker compose up -d && sleep 20 && curl -s localhost:8000/health`
Expected: `{"status":"ok","model_loaded":true,"qdrant_ok":true,"model_version":"baseline-lr-v1"}`

Run:

```bash
curl -s -X POST localhost:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"lyrics": "stadium lights, I am the main event, every seat up, every phone bent"}'
```

Expected: JSON with `mood`, `confidence`, `probabilities` over 5 moods, non-null `explanation`, `model_version: "baseline-lr-v1"`.

Run: `python scripts/index_corpus.py` (host venv, Qdrant still up)
Expected: `indexed 76595 songs into 'songs' at http://localhost:6333` (number ≈ corpus size). Rerun → same count, no duplicates: `curl -s localhost:6333/collections/songs | python -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])"` → same number.

Run: `docker compose down`

- [ ] **Step 3: Commit**

```bash
git add .dockerignore docker/Dockerfile.api docker-compose.yml
git commit -m "feat: add docker compose stack (api + qdrant)"
```

---

### Task 9: Documentation update

**Files:**
- Modify: `CLAUDE.md` (Commands + Architecture sections)
- Modify: `README.md` (Quick Start section)

**Interfaces:**
- Consumes: everything above.
- Produces: docs that match reality.

- [ ] **Step 1: Update CLAUDE.md**

Add to the Commands section of `CLAUDE.md`:

```markdown
# API service (Week-1 industrial elevation — see docs/superpowers/specs/)
pip install -r requirements-dev.txt   # api + test deps
pytest                                 # full test suite (no artifacts needed)
docker compose up                      # qdrant :6333 + api :8000 (needs local models/ + data/)
python scripts/index_corpus.py         # populate qdrant from the processed corpus (idempotent)
uvicorn api.main:app --reload          # api without docker (needs local artifacts + qdrant)
```

Add one paragraph to the Architecture section:

```markdown
**API layer (`api/`)**: FastAPI app factory (`create_app()` in `api/main.py`) with
lifespan-loaded artifacts on `app.state` and DI via `api/deps.py` — tests inject
`FakeMoodModel`/`FakeRetrieval` from `tests/conftest.py` instead of loading pickles.
`api/services/model.py` wraps the baseline behind the `MoodModel` protocol (the Week-2
transformer implements the same protocol). Error contract: `{"error": {code, message}}`
everywhere (`api/errors.py`). Raw lyrics are never logged. Qdrant collection schema
lives in `scripts/index_corpus.py`.
```

- [ ] **Step 2: Update README Quick Start**

In `README.md`, after the existing `streamlit run` line in Quick Start, add:

```markdown
# 4. (new) run the REST API + vector DB
docker compose up            # api on :8000, qdrant on :6333
python scripts/index_corpus.py   # one-time corpus indexing
curl -X POST localhost:8000/v1/predict -H 'content-type: application/json' \
  -d '{"lyrics": "your lyrics here"}'
```

- [ ] **Step 3: Run the full suite one last time and commit**

Run: `pytest`
Expected: all green.

```bash
git add CLAUDE.md README.md
git commit -m "docs: document api service commands and architecture"
```

---

## Out of Scope (later weeks)

- Week 2: Colab fine-tune notebook, `training/evaluate.py` eval harness, MLflow, ONNX export, `models/registry.json`, transformer `MoodModel` implementation.
- Week 3: `GET /v1/search`, `GET /v1/songs`, slowapi rate limiting, `/metrics`, Streamlit rewired as API client.
- Week 4: GitHub Actions CI, HF Spaces deploy, README metrics rewrite, LLM-relabeling stretch.
