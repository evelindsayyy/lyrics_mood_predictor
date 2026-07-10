# LyricMood Week 2 — Transformer Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fine-tuned DistilBERT mood classifier served through the existing `MoodModel` protocol via ONNX CPU inference, selectable per-request alongside the baseline, with an eval harness + MLflow tracking and a Colab-ready training script — everything locally testable before the user runs the real fine-tune on Colab.

**Architecture:** A `models/registry.json` pins which models the API loads and which is default. `TransformerMoodModel` (onnxruntime + tokenizers, no torch at serving time) implements the same `MoodModel` protocol as the baseline; the predict route gains a `model=` query param. `training/` holds the offline side: `finetune_distilbert.py` (Colab GPU; `--smoke` mode runs the full pipeline tiny-and-local) and `evaluate.py` (frozen-split eval harness, markdown + confusion-matrix report, optional MLflow file-backend logging). Tests use a hand-built tiny ONNX graph + trained-in-fixture WordLevel tokenizer — no torch, no downloads.

**Tech Stack:** onnxruntime, tokenizers (serving); torch + transformers + mlflow (training/dev only); shap Text masker for transformer explanations.

**Spec:** `docs/superpowers/specs/2026-07-09-industrial-elevation-design.md` §3.2 + Week-2 row of §7.

## Global Constraints

- Python 3.10+; free-tier only. Serving deps stay torch-free: `requirements-api.txt` may gain `onnxruntime`/`tokenizers` ONLY. torch/transformers/mlflow live in `training/requirements-train.txt` and `requirements-dev.txt`.
- `random_state=42` / `np.random.default_rng(42)` everywhere.
- Every new module docstring carries an AI-attribution block (repo-root-relative depth: `../ATTRIBUTION.md` from `api/` or `training/`, `../../ATTRIBUTION.md` from `api/services/` or `api/routes/`).
- Error envelope `{"error": {"code", "message"}}`; raw lyrics NEVER logged.
- `MoodModel` protocol (`api/services/model.py`) is the contract: `version: str`, `predict(lyrics: str, explain: bool = True) -> PredictionResult`. Do not change `PredictionResult`.
- Explanation failures non-fatal (`explanation=None` + `logger.warning`, no lyrics in logs).
- Existing behavior must not regress: all 33 existing tests keep passing (call-site updates to `create_app` are allowed and specified in Task 3).
- Tests must run WITHOUT heavy artifacts and WITHOUT network (tiny ONNX + in-fixture tokenizer; no HF downloads in tests).
- Rate limiting, `/metrics`, `/v1/search`, `/v1/songs` are Week 3 — do NOT add them.

## File Structure

```
models/registry.json                    # committed; default=baseline until user trains
api/config.py                           # MODIFY: + registry_path
api/services/registry.py                # ModelSpec, Registry, load_registry
api/services/transformer.py             # TransformerMoodModel, load_transformer, softmax
api/services/model.py                   # MODIFY: nothing (protocol untouched)
api/main.py                             # MODIFY: multi-model lifespan via registry
api/deps.py                             # MODIFY: get_models/get_default_model
api/routes/predict.py                   # MODIFY: model= query param
api/routes/health.py                    # MODIFY: loaded-models report
training/__init__.py
training/requirements-train.txt         # torch, transformers, mlflow (Colab/local-smoke)
training/finetune_distilbert.py         # Colab script with --smoke
training/evaluate.py                    # eval harness CLI
training/README.md                      # Colab runbook (Task 7)
tests/unit/test_registry.py
tests/unit/test_transformer_service.py
tests/unit/test_transformer_explain.py
tests/unit/test_evaluate.py
tests/api/test_model_param.py
tests/conftest.py                       # MODIFY: + tiny_onnx_dir fixture
requirements-api.txt                    # MODIFY: + onnxruntime, tokenizers
requirements-dev.txt                    # MODIFY: + onnx, mlflow
```

---

### Task 1: Model registry

**Files:**
- Create: `models/registry.json`, `api/services/registry.py`
- Modify: `api/config.py` (add one field)
- Test: `tests/unit/test_registry.py`

**Interfaces:**
- Consumes: `Settings` (api/config.py), `ArtifactError` (api/services/model.py).
- Produces:
  - `ModelSpec` frozen dataclass: `name: str`, `kind: str` ("baseline" | "onnx"), `version: str`, `dir: Path | None = None`.
  - `Registry` frozen dataclass: `default: str`, `models: dict[str, ModelSpec]`.
  - `load_registry(path: Path) -> Registry` — raises `ArtifactError` (message contains the path) when the file is missing, unparseable, or `default` names an unknown model.
  - `Settings.registry_path: Path = Path("models/registry.json")`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_registry.py`:

```python
"""Tests for api.services.registry."""

import json

import pytest


def _write(tmp_path, payload):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(payload))
    return p


GOOD = {
    "default": "baseline",
    "models": {
        "baseline": {"kind": "baseline", "version": "baseline-lr-v1"},
        "transformer": {"kind": "onnx", "version": "distilbert-mood-v1", "dir": "models/transformer"},
    },
}


def test_load_registry_roundtrip(tmp_path):
    from api.services.registry import load_registry

    reg = load_registry(_write(tmp_path, GOOD))
    assert reg.default == "baseline"
    assert set(reg.models) == {"baseline", "transformer"}
    assert reg.models["transformer"].kind == "onnx"
    assert str(reg.models["transformer"].dir) == "models/transformer"
    assert reg.models["baseline"].dir is None


def test_load_registry_missing_file(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    with pytest.raises(ArtifactError) as exc:
        load_registry(tmp_path / "nope.json")
    assert "nope.json" in str(exc.value)


def test_load_registry_unknown_default(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    bad = {"default": "ghost", "models": {"baseline": {"kind": "baseline", "version": "v"}}}
    with pytest.raises(ArtifactError):
        load_registry(_write(tmp_path, bad))


def test_load_registry_malformed_json(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    p = tmp_path / "registry.json"
    p.write_text("{not json")
    with pytest.raises(ArtifactError):
        load_registry(p)


def test_committed_registry_is_loadable():
    from api.services.registry import load_registry

    reg = load_registry(__import__("pathlib").Path("models/registry.json"))
    assert reg.default == "baseline"
    assert "transformer" in reg.models
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.registry'`

- [ ] **Step 3: Write the implementation**

`models/registry.json` (note: `models/*.json` is NOT gitignored — only `*.pkl/*.npy/*.joblib` are; verify with `git check-ignore models/registry.json`, which must print nothing):

```json
{
  "default": "baseline",
  "models": {
    "baseline": {"kind": "baseline", "version": "baseline-lr-v1"},
    "transformer": {"kind": "onnx", "version": "distilbert-mood-v1", "dir": "models/transformer"}
  }
}
```

`api/services/registry.py`:

```python
"""
Model registry — pins which models the API loads and which is the default.

The registry file is committed (models/registry.json). After training the
transformer on Colab and dropping artifacts into models/transformer/, flip
"default" to "transformer" to promote it.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 registry pinning). See ../../ATTRIBUTION.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from api.services.model import ArtifactError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "baseline" | "onnx"
    version: str
    dir: Path | None = None


@dataclass(frozen=True)
class Registry:
    default: str
    models: dict[str, ModelSpec]


def load_registry(path: Path) -> Registry:
    if not path.exists():
        raise ArtifactError(f"model registry missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = {
            name: ModelSpec(
                name=name,
                kind=spec["kind"],
                version=spec["version"],
                dir=Path(spec["dir"]) if "dir" in spec else None,
            )
            for name, spec in raw["models"].items()
        }
        default = raw["default"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"model registry unreadable: {path} ({exc})") from exc
    if default not in models:
        raise ArtifactError(f"model registry default {default!r} not in models: {path}")
    return Registry(default=default, models=models)
```

`api/config.py` — add one field after `qdrant_collection`:

```python
    registry_path: Path = Path("models/registry.json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_registry.py -v`
Expected: 5 PASSED. Then `pytest` → 38 passed.

- [ ] **Step 5: Commit**

```bash
git add models/registry.json api/services/registry.py api/config.py tests/unit/test_registry.py
git commit -m "feat: add model registry with committed baseline default"
```

---

### Task 2: Transformer serving model (ONNX, no explanation yet)

**Files:**
- Create: `api/services/transformer.py`
- Modify: `requirements-api.txt` (add `onnxruntime>=1.17`, `tokenizers>=0.15`), `requirements-dev.txt` (add `onnx>=1.15`), `tests/conftest.py` (add `tiny_onnx_dir` fixture)
- Test: `tests/unit/test_transformer_service.py`

**Interfaces:**
- Consumes: `PredictionResult`, `ArtifactError` (api/services/model.py).
- Produces:
  - `softmax(x: np.ndarray) -> np.ndarray` (last-axis, numerically stable).
  - `TransformerMoodModel(session, tokenizer, labels: list[str], version: str, max_len: int = 256)` implementing `MoodModel`. `predict(lyrics, explain=True)`: tokenize → `session.run` → softmax → `PredictionResult`. In THIS task `explain=True` returns `explanation=None` (Task 4 adds SHAP); `_predict_proba(texts: list[str]) -> np.ndarray` batch helper (used by Task 4 and Task 5).
  - `load_transformer(model_dir: Path, version: str, max_len: int = 256) -> TransformerMoodModel` — expects `model.onnx`, `tokenizer.json`, `labels.json` in `model_dir`; `ArtifactError` naming the missing file otherwise.
  - conftest fixture `tiny_onnx_dir` (session-scoped tmp dir with the three artifact files) and helper `build_tiny_onnx(vocab_size, n_labels, out_path)`.
- ONNX I/O contract (also binds Task 6's export): inputs `input_ids`, `attention_mask` (int64, [batch, seq]); output `logits` (float32, [batch, n_labels]).

- [ ] **Step 1: Add dependencies**

Append to `requirements-api.txt`:

```
onnxruntime>=1.17
tokenizers>=0.15
```

Append to `requirements-dev.txt`:

```
onnx>=1.15
```

Run: `source .venv/bin/activate && pip install -r requirements-dev.txt`

- [ ] **Step 2: Add the fixture to tests/conftest.py**

Append to `tests/conftest.py`:

```python
def build_tiny_onnx(vocab_size: int, n_labels: int, out_path):
    """Hand-built ONNX graph with the real serving I/O contract:
    Gather(embedding, input_ids) -> ReduceMean(axis=1) -> logits.
    attention_mask is a declared (unused) input so the serving code's feed
    dict matches a real DistilBERT export."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    emb = rng.normal(scale=0.5, size=(vocab_size, n_labels)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gather", ["emb", "input_ids"], ["tok_emb"]),
            helper.make_node("ReduceMean", ["tok_emb"], ["logits"], axes=[1], keepdims=0),
        ],
        name="tiny_mood",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_labels])],
        initializer=[numpy_helper.from_array(emb, name="emb")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


@pytest.fixture(scope="session")
def tiny_onnx_dir(tmp_path_factory):
    """models/transformer-shaped artifact dir: model.onnx + tokenizer.json + labels.json."""
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    d = tmp_path_factory.mktemp("tiny_transformer")

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        [t for t, _ in TINY_SONGS], WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    )
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.enable_truncation(max_length=32)
    tok.save(str(d / "tokenizer.json"))

    labels = ["Angry", "Calm", "Hype", "Romantic", "Sad"]
    (d / "labels.json").write_text(json.dumps(labels))

    build_tiny_onnx(vocab_size=tok.get_vocab_size(), n_labels=len(labels), out_path=d / "model.onnx")
    return d
```

- [ ] **Step 3: Write the failing tests**

`tests/unit/test_transformer_service.py`:

```python
"""Tests for api.services.transformer against the tiny ONNX fixture."""

import numpy as np
import pytest


def _load(tiny_onnx_dir):
    from api.services.transformer import load_transformer

    return load_transformer(tiny_onnx_dir, version="tiny-onnx-v0")


def test_softmax_rows_sum_to_one():
    from api.services.transformer import softmax

    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert not np.isnan(p).any()  # numerically stable at large magnitudes


def test_predict_returns_prediction_result(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("stadium lights bass kicking loud crowd")
    assert r.mood in {"Angry", "Calm", "Hype", "Romantic", "Sad"}
    assert 0.0 < r.confidence <= 1.0
    assert pytest.approx(sum(r.probabilities.values()), abs=1e-5) == 1.0
    assert r.explanation is None  # SHAP lands in the next task
    assert m.version == "tiny-onnx-v0"


def test_predict_is_deterministic(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    a = m.predict("rain empty street coat chair alone")
    b = m.predict("rain empty street coat chair alone")
    assert a.mood == b.mood and a.confidence == b.confidence


def test_predict_proba_batch_shape(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    p = m._predict_proba(["stadium lights", "rain empty street", "tender heart"])
    assert p.shape == (3, 5)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_unknown_words_still_predict(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("zzzz qqqq xxxx")  # all [UNK]
    assert r.mood in {"Angry", "Calm", "Hype", "Romantic", "Sad"}


def test_load_transformer_missing_file(tmp_path):
    from api.services.model import ArtifactError
    from api.services.transformer import load_transformer

    with pytest.raises(ArtifactError) as exc:
        load_transformer(tmp_path, version="v")
    assert "model.onnx" in str(exc.value)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/unit/test_transformer_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.transformer'`

- [ ] **Step 5: Write the implementation**

`api/services/transformer.py`:

```python
"""
Transformer serving model — ONNX CPU inference behind the MoodModel protocol.

Serving never imports torch/transformers: the fine-tuned DistilBERT is
exported to ONNX (training/finetune_distilbert.py) and this module runs it
with onnxruntime + the tokenizers library. Artifact dir contract:
model.onnx, tokenizer.json, labels.json.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2: ONNX int8 serving, no training stack at runtime).
See ../../ATTRIBUTION.md.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from api.services.model import ArtifactError, PredictionResult

REQUIRED_FILES = ("model.onnx", "tokenizer.json", "labels.json")


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class TransformerMoodModel:
    """Fine-tuned transformer via onnxruntime. Implements MoodModel."""

    def __init__(self, session, tokenizer, labels: list[str], version: str, max_len: int = 256):
        self._session = session
        self._tokenizer = tokenizer
        self._labels = list(labels)
        self._max_len = max_len
        self.version = version

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult:
        probs = self._predict_proba([lyrics])[0]
        idx = int(np.argmax(probs))
        return PredictionResult(
            mood=self._labels[idx],
            confidence=float(probs[idx]),
            probabilities={l: float(p) for l, p in zip(self._labels, probs)},
            explanation=None,  # SHAP text explanation added in the explain task
        )

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        max_len = min(self._max_len, max(len(e.ids) for e in encodings))
        ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            n = min(len(enc.ids), max_len)
            ids[i, :n] = enc.ids[:n]
            mask[i, :n] = enc.attention_mask[:n]
        (logits,) = self._session.run(["logits"], {"input_ids": ids, "attention_mask": mask})
        return softmax(np.asarray(logits, dtype=np.float32))


def load_transformer(model_dir: Path, version: str, max_len: int = 256) -> TransformerMoodModel:
    """Load the ONNX artifact dir; fail fast naming the first missing file."""
    model_dir = Path(model_dir)
    for name in REQUIRED_FILES:
        if not (model_dir / name).exists():
            raise ArtifactError(f"transformer artifact missing: {model_dir / name}")
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    return TransformerMoodModel(session, tokenizer, labels, version=version, max_len=max_len)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_transformer_service.py -v`
Expected: 6 PASSED. Then `pytest` → 44 passed.

- [ ] **Step 7: Commit**

```bash
git add requirements-api.txt requirements-dev.txt api/services/transformer.py tests/conftest.py tests/unit/test_transformer_service.py
git commit -m "feat: add onnx transformer serving model behind MoodModel protocol"
```

---

### Task 3: Multi-model serving (registry-driven lifespan + model= param)

**Files:**
- Modify: `api/main.py`, `api/deps.py`, `api/routes/predict.py`, `api/routes/health.py`
- Modify: `tests/api/test_health.py`, `tests/api/test_predict.py` (call-site update: `model=` → `models=`/`default=`)
- Test: `tests/api/test_model_param.py`

**Interfaces:**
- Consumes: `Registry`/`load_registry` (Task 1), `load_transformer` (Task 2), `load_baseline`, fakes from conftest.
- Produces:
  - `create_app(settings=None, models: dict[str, MoodModel] | None = None, default: str | None = None, retrieval=None)`. Injected `models`+`default` skip all real loading (same eager pattern as before). Real path (lifespan): `load_registry(cfg.registry_path)`; for each spec — kind "baseline" → `load_baseline(cfg)`; kind "onnx" → `load_transformer(spec.dir, spec.version)` ONLY if `spec.dir` exists on disk (skip silently with a `logger.info("model_unavailable", model=name)` otherwise). The registry DEFAULT must load or startup raises `ArtifactError`.
  - `api/deps.py`: `get_models(request) -> dict[str, MoodModel]`, `get_default_model_name(request) -> str` (replaces `get_model`; keep `get_retrieval`).
  - Predict route: `model: str | None = Query(None)` → name = `model or default`; name not in registry-loaded dict AND not a registry name → 400 `unknown_model`; registry name whose artifacts weren't loaded → 503 `model_unavailable`. Response `model_version` comes from the chosen model.
  - Health: `{"status", "model_loaded", "qdrant_ok", "model_version", "models_loaded": [names], "default_model": name}` — `model_version` stays the DEFAULT model's version (backward compatible).
  - App state: `app.state.models: dict[str, MoodModel]`, `app.state.default_model: str`, `app.state.registry_names: set[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/api/test_model_param.py`:

```python
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
```

Also update existing call sites (mechanical, same assertion bodies):
- In `tests/api/test_health.py` `_client()`: `create_app(model=FakeMoodModel(), retrieval=...)` → `create_app(models={"baseline": FakeMoodModel()}, default="baseline", retrieval=...)`.
- In `tests/api/test_predict.py` `_client()`: same substitution (`model=model or FakeMoodModel()` → `models={"baseline": model or FakeMoodModel()}, default="baseline"`).
- In `tests/api/test_health.py::test_health_ok`, the expected JSON gains the two new keys: `"models_loaded": ["baseline"], "default_model": "baseline"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/api/test_model_param.py -v`
Expected: FAIL with `TypeError: create_app() got an unexpected keyword argument 'models'`

- [ ] **Step 3: Implement**

`api/deps.py` (full new content; keep the attribution block added in week 1, updating the described accessors):

```python
"""Dependency accessors — routes depend on app.state, tests inject fakes.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

from fastapi import Request

from api.services.model import MoodModel
from api.services.retrieval import RetrievalClient


def get_models(request: Request) -> dict[str, MoodModel]:
    return request.app.state.models


def get_default_model_name(request: Request) -> str:
    return request.app.state.default_model


def get_retrieval(request: Request) -> RetrievalClient:
    return request.app.state.retrieval
```

`api/main.py` — replace the model-loading parts (docstring + attribution stays; imports gain `registry`/`transformer`):

```python
def create_app(
    settings: Settings | None = None,
    models: dict[str, MoodModel] | None = None,
    default: str | None = None,
    retrieval: RetrievalClient | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "models"):
            reg = load_registry(cfg.registry_path)
            loaded: dict[str, MoodModel] = {}
            for name, spec in reg.models.items():
                if spec.kind == "baseline":
                    loaded[name] = load_baseline(cfg)
                elif spec.kind == "onnx" and spec.dir is not None and spec.dir.exists():
                    loaded[name] = load_transformer(spec.dir, spec.version)
                else:
                    logger.info("model_unavailable", model=name)
            if reg.default not in loaded:
                raise ArtifactError(f"registry default {reg.default!r} failed to load")
            app.state.models = loaded
            app.state.default_model = reg.default
            app.state.registry_names = set(reg.models)
        if not hasattr(app.state, "retrieval"):
            app.state.retrieval = QdrantRetrieval(cfg.qdrant_url)
        yield

    app = FastAPI(title="LyricMood API", version="1.0", lifespan=lifespan)
    app.state.settings = cfg
    if models is not None:
        app.state.models = dict(models)
        app.state.default_model = default if default is not None else next(iter(models))
        app.state.registry_names = set(models)
    if retrieval is not None:
        app.state.retrieval = retrieval
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    return app
```

(Module-level `logger = structlog.get_logger()` and imports: `from api.services.registry import load_registry`, `from api.services.transformer import load_transformer`, `from api.services.model import ArtifactError, MoodModel, load_baseline`.)

`api/routes/predict.py` — signature and selection replace the single-model dependency (body of the happy path unchanged):

```python
@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    explain: bool = Query(True),
    model: str | None = Query(None),
    models: dict[str, MoodModel] = Depends(get_models),
    default_name: str = Depends(get_default_model_name),
    request: Request = None,
) -> PredictResponse:
    text = req.lyrics.strip()
    if not text:
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")

    name = model or default_name
    if name not in models:
        registry_names = getattr(request.app.state, "registry_names", set(models))
        if name in registry_names:
            raise ApiError(503, "model_unavailable", f"model {name!r} is registered but not loaded")
        raise ApiError(400, "unknown_model", f"unknown model {name!r}")

    chosen = models[name]
    result = chosen.predict(text, explain=explain)
    logger.info("predict", input_chars=len(text), mood=result.mood, model=chosen.version)

    return PredictResponse(
        mood=result.mood,
        confidence=result.confidence,
        probabilities=result.probabilities,
        explanation=None
        if result.explanation is None
        else [TokenWeight(token=t, weight=w) for t, w in result.explanation],
        model_version=chosen.version,
        warnings=non_english_warnings(text),
    )
```

(Imports switch from `get_model` to `get_models, get_default_model_name`; add `Request` from fastapi.)

`api/routes/health.py`:

```python
@router.get("/health")
def health(
    models: dict[str, MoodModel] = Depends(get_models),
    default_name: str = Depends(get_default_model_name),
    retrieval: RetrievalClient = Depends(get_retrieval),
):
    return {
        "status": "ok",
        "model_loaded": default_name in models,
        "qdrant_ok": retrieval.ping(),
        "model_version": models[default_name].version,
        "models_loaded": sorted(models),
        "default_model": default_name,
    }
```

- [ ] **Step 4: Run the whole suite**

Run: `pytest`
Expected: all pass (44 prior + 5 new = 49; the updated health/predict call sites keep their assertions).

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/deps.py api/routes/predict.py api/routes/health.py tests/api/test_model_param.py tests/api/test_health.py tests/api/test_predict.py
git commit -m "feat: registry-driven multi-model serving with model= param"
```

---

### Task 4: Transformer explanations (SHAP Text masker)

**Files:**
- Modify: `api/services/transformer.py`
- Test: `tests/unit/test_transformer_explain.py`

**Interfaces:**
- Consumes: `TransformerMoodModel._predict_proba` (Task 2).
- Produces: `TransformerMoodModel.predict(..., explain=True)` now returns `explanation: list[tuple[str, float]]` — top-10 tokens of the predicted class by |SHAP|, sorted signed descending — or `None` on any failure (warning-logged, no lyrics). New constructor knobs with defaults: `explain_max_chars: int = 300`, `explain_max_evals: int = 64`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_transformer_explain.py`:

```python
"""Tests for transformer SHAP text explanations on the tiny ONNX model."""


def _load(tiny_onnx_dir):
    from api.services.transformer import load_transformer

    return load_transformer(tiny_onnx_dir, version="tiny-onnx-v0")


def test_explanation_returns_token_weights(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("stadium lights bass kicking loud crowd", explain=True)
    assert r.explanation is not None
    assert 1 <= len(r.explanation) <= 10
    for token, weight in r.explanation:
        assert isinstance(token, str) and token.strip()
        assert isinstance(weight, float)
    # sorted signed descending
    weights = [w for _, w in r.explanation]
    assert weights == sorted(weights, reverse=True)


def test_explain_false_skips(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    assert m.predict("stadium lights", explain=False).explanation is None


def test_explanation_failure_is_non_fatal(tiny_onnx_dir, monkeypatch):
    m = _load(tiny_onnx_dir)
    monkeypatch.setattr(m, "_explain", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    # predict must catch and degrade, not raise
    r = m.predict("stadium lights", explain=True)
    assert r.explanation is None


def test_long_input_is_capped(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    long_text = "stadium lights bass " * 200  # ~4000 chars
    r = m.predict(long_text, explain=True)
    # must complete (input capped to explain_max_chars) and stay bounded
    assert r.explanation is None or len(r.explanation) <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_transformer_explain.py -v`
Expected: `test_explanation_returns_token_weights` FAILS (explanation is None); others may pass.

- [ ] **Step 3: Implement**

In `api/services/transformer.py`: add `import re`, `import structlog`, module `logger = structlog.get_logger()`; extend `__init__` with `explain_max_chars: int = 300, explain_max_evals: int = 64` (store on self); change `predict` and add `_explain`:

```python
    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult:
        probs = self._predict_proba([lyrics])[0]
        idx = int(np.argmax(probs))
        explanation = None
        if explain:
            try:
                explanation = self._explain(lyrics)
            except Exception as exc:
                logger.warning("explain_failed", error=type(exc).__name__)
        return PredictionResult(
            mood=self._labels[idx],
            confidence=float(probs[idx]),
            probabilities={l: float(p) for l, p in zip(self._labels, probs)},
            explanation=explanation,
        )

    def _explain(self, lyrics: str) -> list[tuple[str, float]] | None:
        """Token-level SHAP via the Text masker; capped for latency."""
        import shap  # local import: keeps module import light

        text = lyrics[: self._explain_max_chars]
        probs = self._predict_proba([text])[0]
        class_idx = int(np.argmax(probs))

        masker = shap.maskers.Text(r"\W+")  # regex splitter — tokenizer-agnostic
        explainer = shap.Explainer(
            lambda texts: self._predict_proba(list(texts)), masker, silent=True
        )
        exp = explainer([text], max_evals=self._explain_max_evals)
        tokens = [str(t).strip() for t in exp.data[0]]
        values = np.asarray(exp.values[0])[:, class_idx]

        pairs = [(t, float(v)) for t, v in zip(tokens, values) if t]
        if not pairs:
            return None
        top = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:10]
        return sorted(top, key=lambda kv: kv[1], reverse=True)
```

NOTE for the implementer: shap's Partition explainer requires `max_evals >= 2 * num_tokens + 1` for exact mode but accepts smaller budgets with approximation; if the installed shap version raises on small `max_evals`, raise the cap passed to `explainer(...)` to `max(self._explain_max_evals, 2 * len(text.split()) + 1)` and note it in your report. Keep the wall-clock bounded — the test with ~4000 chars must finish in seconds because of the char cap.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_transformer_explain.py -v`
Expected: 4 PASSED. Then `pytest` → 53 passed.

- [ ] **Step 5: Commit**

```bash
git add api/services/transformer.py tests/unit/test_transformer_explain.py
git commit -m "feat: add capped shap text explanations to transformer model"
```

---

### Task 5: Eval harness (training/evaluate.py)

**Files:**
- Create: `training/__init__.py` (empty), `training/evaluate.py`
- Modify: `requirements-dev.txt` (add `mlflow>=2.10`)
- Test: `tests/unit/test_evaluate.py`

**Interfaces:**
- Consumes: `src.classify.split_data` (existing, unchanged), `MoodModel` protocol, `load_baseline`/`load_transformer`/`load_registry`.
- Produces:
  - `frozen_test_split(df: pd.DataFrame) -> pd.DataFrame` — reproduces the SAME test rows as the notebooks: `split_data(df.index.to_numpy(), df["mood"], random_state=42)` and returns `df.loc[X_test]`.
  - `evaluate_predictor(model: MoodModel, df, text_col="lyrics", label_col="mood", limit: int | None = None) -> dict` — keys: `accuracy`, `macro_f1`, `per_class_precision`, `per_class_recall`, `confusion` (nested dict true→pred→count), `n`, `model_version`. Iterates `model.predict(text, explain=False)`.
  - `majority_baseline_macro_f1(y) -> float`.
  - `passes_quality_gate(results, y) -> bool` — `results["macro_f1"] > majority_baseline_macro_f1(y)`.
  - `write_report(results: dict, name: str, out_dir: Path = Path("results")) -> Path` — writes `results/eval_<name>.md` (metric table + confusion matrix as a markdown table).
  - `log_mlflow(results: dict, name: str, params: dict) -> None` — no-op with a warning if mlflow isn't importable; else logs params/metrics to `./mlruns` under experiment "lyricmood".
  - CLI: `python -m training.evaluate --model baseline|transformer [--limit N] [--no-mlflow]` — loads the registry + artifacts, prints the metric summary, writes the report, exits 1 if the quality gate fails.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_evaluate.py`:

```python
"""Tests for the eval harness against the tiny fixtures — no artifacts, no mlflow."""

import pandas as pd
import pytest

from tests.conftest import TINY_SONGS, FakeMoodModel


@pytest.fixture
def tiny_df():
    # x4 so the stratified 10% test split has >= 1 row per class (sklearn
    # raises if n_test < n_classes on a 15-row frame)
    songs = TINY_SONGS * 4
    return pd.DataFrame(
        {"lyrics": [t for t, _ in songs], "mood": [m for _, m in songs]}
    )


def test_frozen_test_split_is_deterministic(tiny_df):
    from training.evaluate import frozen_test_split

    a = frozen_test_split(tiny_df)
    b = frozen_test_split(tiny_df)
    assert list(a.index) == list(b.index)
    assert 0 < len(a) < len(tiny_df)


def test_evaluate_predictor_perfect_oracle(tiny_df):
    from training.evaluate import evaluate_predictor

    class Oracle:
        version = "oracle-v0"

        def predict(self, lyrics, explain=True):
            from api.services.model import PredictionResult

            mood = dict(TINY_SONGS)[lyrics]
            return PredictionResult(mood=mood, confidence=1.0, probabilities={mood: 1.0}, explanation=None)

    r = evaluate_predictor(Oracle(), tiny_df)
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["n"] == len(tiny_df)
    assert r["model_version"] == "oracle-v0"


def test_evaluate_predictor_constant_model_and_gate(tiny_df):
    from training.evaluate import evaluate_predictor, passes_quality_gate

    r = evaluate_predictor(FakeMoodModel(mood="Hype"), tiny_df)
    assert r["accuracy"] == pytest.approx(3 / 15)  # 12/60 — same ratio as 3/15
    # constant predictor == majority baseline → does NOT beat it
    assert passes_quality_gate(r, tiny_df["mood"]) is False


def test_evaluate_limit(tiny_df):
    from training.evaluate import evaluate_predictor

    r = evaluate_predictor(FakeMoodModel(), tiny_df, limit=5)
    assert r["n"] == 5


def test_write_report(tmp_path, tiny_df):
    from training.evaluate import evaluate_predictor, write_report

    r = evaluate_predictor(FakeMoodModel(mood="Hype"), tiny_df)
    path = write_report(r, "fake", out_dir=tmp_path)
    text = path.read_text()
    assert "macro_f1" in text and "Hype" in text
    assert path.name == "eval_fake.md"


def test_log_mlflow_noop_without_mlflow(monkeypatch, tiny_df):
    import builtins

    from training.evaluate import evaluate_predictor, log_mlflow

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = evaluate_predictor(FakeMoodModel(), tiny_df)
    log_mlflow(r, "fake", params={"x": 1})  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Implement**

Append `mlflow>=2.10` to `requirements-dev.txt`; `pip install -r requirements-dev.txt`.

`training/__init__.py`: empty.

`training/evaluate.py`:

```python
"""
Eval harness — one command to score any registered model on the frozen test
split, write a markdown report, and (optionally) log the run to MLflow.

The split is byte-identical to the notebooks' split: same
src.classify.split_data, same random_state=42, stratified on mood.

Usage:
    python -m training.evaluate --model baseline
    python -m training.evaluate --model transformer --limit 500 --no-mlflow

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 eval harness + quality gate). See ../ATTRIBUTION.md.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classify import split_data  # noqa: E402


def frozen_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """The exact test rows the notebooks used (random_state=42, stratified)."""
    idx = df.index.to_numpy()
    _, _, X_test, _, _, _ = split_data(idx, df["mood"], random_state=42)
    return df.loc[X_test]


def evaluate_predictor(model, df: pd.DataFrame, text_col: str = "lyrics",
                       label_col: str = "mood", limit: int | None = None) -> dict:
    rows = df if limit is None else df.iloc[:limit]
    y_true = rows[label_col].tolist()
    y_pred = [model.predict(t, explain=False).mood for t in rows[text_col]]

    classes = sorted(set(y_true) | set(y_pred))
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, pr in zip(y_true, y_pred):
        confusion[t][pr] += 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_precision": {c: float(v) for c, v in zip(classes, p)},
        "per_class_recall": {c: float(v) for c, v in zip(classes, r)},
        "confusion": {t: dict(d) for t, d in confusion.items()},
        "n": len(rows),
        "model_version": model.version,
    }


def majority_baseline_macro_f1(y) -> float:
    majority = Counter(y).most_common(1)[0][0]
    return float(f1_score(list(y), [majority] * len(y), average="macro"))


def passes_quality_gate(results: dict, y) -> bool:
    return results["macro_f1"] > majority_baseline_macro_f1(y)


def write_report(results: dict, name: str, out_dir: Path = Path("results")) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(results["per_class_precision"])

    lines = [
        f"# Eval report — {name} ({results['model_version']})",
        "",
        f"- n: {results['n']}",
        f"- accuracy: {results['accuracy']:.4f}",
        f"- macro_f1: {results['macro_f1']:.4f}",
        "",
        "| class | precision | recall |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c} | {results['per_class_precision'][c]:.3f} | {results['per_class_recall'][c]:.3f} |"
        for c in classes
    ]
    lines += ["", "## Confusion (true → predicted)", "", "| true \\ pred | " + " | ".join(classes) + " |",
              "|---" * (len(classes) + 1) + "|"]
    for t in classes:
        row = results["confusion"].get(t, {})
        lines.append(f"| {t} | " + " | ".join(str(row.get(c, 0)) for c in classes) + " |")

    path = out_dir / f"eval_{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def log_mlflow(results: dict, name: str, params: dict) -> None:
    """Log to ./mlruns (file backend). Silently skips if mlflow isn't installed."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed — skipping tracking")
        return
    mlflow.set_experiment("lyricmood")
    with mlflow.start_run(run_name=f"eval-{name}"):
        mlflow.log_params({**params, "model_version": results["model_version"]})
        mlflow.log_metrics(
            {"accuracy": results["accuracy"], "macro_f1": results["macro_f1"], "n": results["n"]}
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a registered model on the frozen test split")
    parser.add_argument("--model", required=True, choices=["baseline", "transformer"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    from api.config import Settings
    from api.services.model import load_baseline
    from api.services.registry import load_registry
    from api.services.transformer import load_transformer

    settings = Settings()
    reg = load_registry(settings.registry_path)
    spec = reg.models[args.model]
    model = load_baseline(settings) if spec.kind == "baseline" else load_transformer(spec.dir, spec.version)

    df = pd.read_csv(settings.labeled_songs_path)
    test_df = frozen_test_split(df)
    results = evaluate_predictor(model, test_df, limit=args.limit)

    print(f"model={args.model} version={results['model_version']} n={results['n']}")
    print(f"accuracy={results['accuracy']:.4f} macro_f1={results['macro_f1']:.4f}")
    report = write_report(results, args.model)
    print(f"report: {report}")

    if not args.no_mlflow:
        log_mlflow(results, args.model, params={"model": args.model, "limit": args.limit or 0})

    if not passes_quality_gate(results, test_df["mood"].iloc[: args.limit or len(test_df)]):
        print("QUALITY GATE FAILED: model does not beat majority-class macro F1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evaluate.py -v`
Expected: 6 PASSED. Then `pytest` → 59 passed.

- [ ] **Step 5: Real-artifact sanity run (local artifacts exist)**

Run: `.venv/bin/python -m training.evaluate --model baseline --limit 300 --no-mlflow`
Expected: prints accuracy/macro_f1 in the neighborhood of the README numbers (limit-subset noise is fine), writes `results/eval_baseline.md`, exits 0. Paste the output in your report. Do NOT commit `results/eval_baseline.md` (results/*.md isn't gitignored, but generated eval reports from partial runs shouldn't be committed — `git status` before committing).

- [ ] **Step 6: Commit**

```bash
git add training/__init__.py training/evaluate.py requirements-dev.txt tests/unit/test_evaluate.py
git commit -m "feat: add eval harness with frozen split, quality gate, mlflow logging"
```

---

### Task 6: Colab fine-tune script (with local --smoke mode)

**Files:**
- Create: `training/requirements-train.txt`, `training/finetune_distilbert.py`
- Test: smoke run (no pytest file — this script needs torch; it is validated by executing `--smoke` locally, which IS the test)

**Interfaces:**
- Consumes: `data/processed/songs_labeled.csv` schema (`lyrics`, `mood` columns), `frozen_test_split`/`split_data` convention (random_state=42), ONNX I/O contract from Task 2 (`input_ids`+`attention_mask` → `logits`), artifact-dir contract (`model.onnx`, `tokenizer.json`, `labels.json`).
- Produces: `python training/finetune_distilbert.py --data <csv> --out models/transformer [--smoke]` → writes the three artifact files + `metrics.json` (best val macro F1, epochs run) to `--out`. `--smoke`: tiny random-init DistilBERT config (2 layers, hidden 64, heads 2), ≤64 rows, 1 epoch, CPU — proves the full pipeline (load → split → tokenize → train → select → export → quantize) in under ~3 minutes with no GPU.

- [ ] **Step 1: Write the requirements and script**

`training/requirements-train.txt`:

```
torch>=2.1
transformers>=4.38
onnx>=1.15
onnxruntime>=1.17
pandas>=2.0
scikit-learn>=1.3
mlflow>=2.10
```

`training/finetune_distilbert.py`:

```python
"""
Fine-tune DistilBERT for 5-mood classification and export to quantized ONNX.

Designed for Colab free GPU (full mode, ~1-2h) but fully testable locally:
--smoke runs the identical pipeline with a tiny random-init config on CPU in
minutes. Artifacts land in --out as model.onnx + tokenizer.json + labels.json
(+ metrics.json), the exact contract api/services/transformer.py loads.

Split discipline: identical to the notebooks — src.classify.split_data,
random_state=42, stratified. The test split is NEVER touched here; final test
metrics come from training/evaluate.py only.

Usage (Colab):   python training/finetune_distilbert.py --data data/processed/songs_labeled.csv --out models/transformer
Usage (smoke):   python training/finetune_distilbert.py --data data/processed/songs_labeled.csv --out /tmp/smoke_transformer --smoke

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 training recipe: max_len 256, 2-3 epochs, class-weighted
loss, early stopping on val macro F1, int8 dynamic quantization).
See ../ATTRIBUTION.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_LEN = 256
MODEL_NAME = "distilbert-base-uncased"


def load_splits(data_path: str):
    from src.classify import split_data

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["lyrics", "mood"]).reset_index(drop=True)
    idx = df.index.to_numpy()
    X_tr, X_val, _, _, _, _ = split_data(idx, df["mood"], random_state=42)
    return df.loc[X_tr], df.loc[X_val]


def class_weights(y: pd.Series, labels: list[str]) -> np.ndarray:
    counts = y.value_counts()
    n, k = len(y), len(labels)
    return np.array([n / (k * counts[l]) for l in labels], dtype=np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny model, tiny data, CPU, minutes")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_splits(args.data)
    if args.smoke:
        train_df, val_df = train_df.iloc[:48], val_df.iloc[:16]
        args.epochs, args.batch_size = 1, 8

    labels = sorted(train_df["mood"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.smoke:
        config = DistilBertConfig(
            vocab_size=tokenizer.vocab_size, n_layers=2, dim=64, n_heads=2,
            hidden_dim=128, num_labels=len(labels), max_position_embeddings=512,
        )
        model = DistilBertForSequenceClassification(config)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    model.to(device)

    def encode(df: pd.DataFrame) -> TensorDataset:
        enc = tokenizer(
            df["lyrics"].astype(str).tolist(), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        y = torch.tensor([label_to_id[m] for m in df["mood"]], dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_dl = DataLoader(encode(train_df), batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
    val_dl = DataLoader(encode(val_df), batch_size=args.batch_size)

    weights = torch.tensor(class_weights(train_df["mood"], labels)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def val_macro_f1() -> float:
        from sklearn.metrics import f1_score

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, y in val_dl:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                preds += logits.argmax(-1).cpu().tolist()
                trues += y.tolist()
        return float(f1_score(trues, preds, average="macro"))

    best_f1, best_state, epochs_run = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_dl):
            optim.zero_grad()
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optim.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        f1 = val_macro_f1()
        epochs_run = epoch + 1
        print(f"epoch {epoch}: val macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print("early stop: val macro F1 did not improve")
            break

    model.load_state_dict(best_state)
    model.cpu().eval()

    # --- export: ONNX with the serving I/O contract, then int8 dynamic quantization
    dummy = tokenizer(["dummy lyrics for export"], return_tensors="pt", padding=True)
    model.config.return_dict = False  # tuple outputs so torch.onnx.export maps output_names cleanly
    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(out_dir / "model.onnx"), weight_type=QuantType.QInt8)
    fp32_path.unlink()

    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "labels.json").write_text(json.dumps(labels))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"best_val_macro_f1": best_f1, "epochs_run": epochs_run, "smoke": args.smoke}
    ))

    print(f"artifacts written to {out_dir} (best val macro F1: {best_f1:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Install training deps and smoke-run**

Run: `pip install -r training/requirements-train.txt` (torch CPU wheel — a few hundred MB; report if disk/network makes this impractical and mark the smoke as blocked rather than faking it).

Run:

```bash
.venv/bin/python training/finetune_distilbert.py \
  --data data/processed/songs_labeled.csv --out /tmp/smoke_transformer --smoke
```

Expected: completes in minutes on CPU; `/tmp/smoke_transformer/` contains `model.onnx`, `tokenizer.json`, `labels.json`, `metrics.json`.

- [ ] **Step 3: Prove the smoke artifacts load in the REAL serving path**

Run:

```bash
.venv/bin/python -c "
from api.services.transformer import load_transformer
m = load_transformer(__import__('pathlib').Path('/tmp/smoke_transformer'), version='smoke-v0')
r = m.predict('stadium lights bass kicking loud crowd', explain=False)
print(r.mood, round(r.confidence, 3), sorted(r.probabilities))
"
```

Expected: prints a mood from the 5 labels + probabilities over 5 classes. This closes the loop: training-script output → serving loader, no torch at load time. Paste output in your report.

- [ ] **Step 4: Full suite + commit**

Run: `pytest`
Expected: 59 passed (script has no pytest file; the smoke IS its verification).

```bash
git add training/requirements-train.txt training/finetune_distilbert.py
git commit -m "feat: add colab fine-tune script with local smoke mode and onnx export"
```

---

### Task 7: Colab runbook + docs

**Files:**
- Create: `training/README.md`
- Modify: `README.md` (evaluation section note), `CLAUDE.md` (commands + architecture), `ATTRIBUTION.md` (training/ entry)

**Interfaces:** consumes everything; produces the user-facing handoff.

- [ ] **Step 1: Write training/README.md**

```markdown
# Training — fine-tune DistilBERT on Colab (free tier)

The serving stack ships with the TF-IDF baseline as default. This runbook
produces `models/transformer/` artifacts and promotes them.

## 1. One-time smoke check (local, no GPU)

    pip install -r training/requirements-train.txt
    python training/finetune_distilbert.py --data data/processed/songs_labeled.csv --out /tmp/smoke_transformer --smoke

Should finish in minutes and print a best-val-F1 line. This validates the
pipeline, not the model quality.

## 2. Real fine-tune (Colab free GPU, ~1-2 h)

1. Upload `data/processed/songs_labeled.csv` to your Google Drive.
2. New Colab notebook → Runtime → Change runtime type → T4 GPU.
3. Cells:

       from google.colab import drive; drive.mount('/content/drive')
       !git clone https://github.com/<you>/LyricsMoodPredictor && cd LyricsMoodPredictor
       %cd LyricsMoodPredictor
       !pip install -q -r training/requirements-train.txt
       !python training/finetune_distilbert.py \
           --data /content/drive/MyDrive/songs_labeled.csv \
           --out /content/drive/MyDrive/lyricmood_transformer

4. When it finishes, download the four files from
   `Drive/lyricmood_transformer/` into `models/transformer/` locally.

## 3. Evaluate + promote

    python -m training.evaluate --model transformer            # frozen test split, writes results/eval_transformer.md
    python -m training.evaluate --model baseline               # regenerate the comparison row

If the transformer clears the gate and beats the baseline's macro F1:
edit `models/registry.json` → `"default": "transformer"`, restart the API
(`docker compose up --build`), and update the README comparison table from
the two eval reports.

MLflow runs land in `./mlruns` — `mlflow ui` to browse.
```

- [ ] **Step 2: Update repo docs**

`CLAUDE.md` — in the API commands block, add:

```markdown
python -m training.evaluate --model baseline   # eval harness (frozen split, quality gate, results/eval_*.md)
python training/finetune_distilbert.py --smoke --data data/processed/songs_labeled.csv --out /tmp/smoke  # pipeline smoke
```

and in the Architecture API paragraph, append:

```markdown
Multi-model serving is registry-driven: `models/registry.json` pins loaded models +
default; `POST /v1/predict?model=baseline|transformer` selects per-request
(400 unknown_model / 503 model_unavailable). The transformer path is
`training/finetune_distilbert.py` (Colab) → ONNX int8 → `api/services/transformer.py`
(onnxruntime, torch-free). Eval: `training/evaluate.py` — same frozen split as the
notebooks (random_state=42).
```

`README.md` — under the Evaluation section heading, add one line:

```markdown
> **Week 2 (in progress):** a fine-tuned DistilBERT is being added behind the same API
> (`?model=transformer`). Numbers will appear here from `training/evaluate.py` reports
> once the Colab fine-tune completes — see [training/README.md](training/README.md).
```

`ATTRIBUTION.md` — extend the api/ section entry (or add a sibling) covering `training/` and `api/services/{registry,transformer}.py`, same style as the Week-1 entry.

- [ ] **Step 3: Full suite + commit**

Run: `pytest` → 59 passed.

```bash
git add training/README.md README.md ATTRIBUTION.md
git commit -m "docs: add colab fine-tune runbook and week-2 architecture docs"
```

(CLAUDE.md is intentionally untracked — edit on disk, do not force-add.)

---

## Out of Scope (Week 3-4)

`/v1/search`, `/v1/songs`, rate limiting, `/metrics`, Streamlit-as-client, CI, HF Spaces deploy, README metric-table update (blocked on the user's real Colab run).

## Handoff After This Plan

The user runs `training/README.md` §2 on Colab, drops artifacts into `models/transformer/`, runs the two evaluate commands, and flips the registry default if the transformer wins. Everything else is already wired.
