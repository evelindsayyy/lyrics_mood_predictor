# LyricMood Week 4 — Ship It Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CI on every push (lint + tests + docker build), a single-container HF Spaces demo package (API + UI + embedded file-based Qdrant, verified locally, deploy runbook for the user), the final README rewrite with architecture diagram and CI badge, and retirement of the deferred-minors backlog.

**Architecture:** The demo container runs uvicorn (:8000, internal) + Streamlit (:7860, the Space's port) via a launcher script, with a `demo/` bundle holding all model artifacts and a **local-path Qdrant** (`QdrantClient(path=...)` — no server process) built from the corpus with excerpt-only payloads. Full lyrics (`songs_labeled.csv`) are deliberately NOT bundled (copyright + 150MB): the lyrics store degrades, so `/v1/songs` single-match analysis 503s on the demo while predict/search/similar/candidates all work — the API's existing degraded-mode contracts make this free.

**Tech Stack:** GitHub Actions, ruff, git-lfs (runbook only), qdrant-client local mode.

**Spec:** `docs/superpowers/specs/2026-07-09-industrial-elevation-design.md` Week-4 row of §7 + §9 risk row "HF Spaces single container mode". The LLM-relabeling stretch is OUT (documented as future work in the README).

## Global Constraints

- Free tier only. Tests stay artifact-free and network-free; suite baseline 101 passed — every task states its expected count.
- Error envelope, no-raw-lyrics logging, torch-free serving, `random_state=42`, AI-attribution blocks: all week 1–3 invariants hold.
- `requirements-api.txt` gains NOTHING this week. `ruff` goes in `requirements-dev.txt`.
- Full lyrics never enter the demo bundle or any public artifact — excerpts (≤300 chars, already in Qdrant payloads) only.
- CLAUDE.md is edited on disk only (untracked by repo convention).
- Deferred-minors cleanup (Task 1) changes NO public contracts — response shapes, routes, and settings names stay identical (new optional setting `qdrant_path` in Task 3 is additive).

## File Structure

```
api/services/embedder.py        # MODIFY: embed([]) guard
api/services/registry.py        # MODIFY: kind validation
api/services/retrieval.py       # MODIFY: ratio-asymmetry comment, QdrantRetrieval.local()
api/services/transformer.py     # MODIFY: E741 rename
api/services/model.py           # MODIFY: E741 rename (if present)
api/routes/search.py            # MODIFY: whitespace-only q guard
api/main.py                     # MODIFY: single count() in skew helper; local-qdrant wiring
api/config.py                   # MODIFY: qdrant_path setting
training/evaluate.py            # MODIFY: --limit validation, unused import
scripts/index_corpus.py         # MODIFY: --local-path arg
scripts/build_demo_bundle.py    # assembles demo/ (models + local qdrant)
docker/Dockerfile.spaces        # single-container demo image
docker/spaces_launcher.sh       # uvicorn + streamlit supervisor
docs/DEPLOY_SPACES.md           # user runbook (HF account required)
.github/workflows/ci.yml        # lint + test + docker-build jobs
ruff.toml
README.md                       # REWRITE (Task 5)
tests/...                       # per-task additions
```

---

### Task 1: Deferred-minors cleanup batch

**Files:**
- Modify: `api/main.py`, `api/services/embedder.py`, `api/services/registry.py`, `api/services/retrieval.py`, `api/routes/search.py`, `training/evaluate.py`, `tests/conftest.py`, `tests/unit/test_transformer_service.py`
- Test: extend `tests/unit/test_embedder.py`, `tests/unit/test_registry.py`, `tests/api/test_search.py`

**Interfaces:** no public contract changes; internal fixes only.

- [ ] **Step 1: Write the failing tests (three additions)**

Append to `tests/unit/test_embedder.py`:

```python
def test_embed_empty_list_raises(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    with pytest.raises(ValueError, match="non-empty"):
        e.embed([])
```

Append to `tests/unit/test_registry.py`:

```python
def test_load_registry_unknown_kind(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    bad = {"default": "m", "models": {"m": {"kind": "quantum", "version": "v"}}}
    with pytest.raises(ArtifactError, match="kind"):
        load_registry(_write(tmp_path, bad))
```

Append to `tests/api/test_search.py`:

```python
def test_search_whitespace_only_q_400():
    r = _client().get("/v1/search?q=%20%20%20%20")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_query"
```

Run each new test → FAIL (ValueError not raised / no kind validation / 200-or-503 instead of 400).

- [ ] **Step 2: Implement the batch**

1. `api/services/embedder.py` — top of `embed`: `if not texts: raise ValueError("texts must be non-empty")`.
2. `api/services/registry.py` — in `load_registry`'s parse block, after building each spec: `if spec.kind not in ("baseline", "onnx"): raise ArtifactError(f"model registry unknown kind {spec.kind!r}: {path}")` (implement however fits the existing loop; message must contain "kind").
3. `api/routes/search.py` — in `search()` before `_validate_mood`: `if not q.strip(): raise ApiError(400, "empty_query", "q must contain non-whitespace text")`.
4. `api/main.py` — rework the skew helper so `retrieval.count()` is called EXACTLY once per startup check: the helper computes the count in a try/except, logs the `lyrics_index_skew` warning itself (with both numbers) on mismatch, and returns bool; the lifespan only uses the bool and no longer logs/counts. Keep the existing test `test_skew_check_disables_store` passing unchanged.
5. `api/services/retrieval.py` — add a 2-line comment above `find_song`'s primary-path return explaining why MatchText hits are unfloored while the fallback gates at 0.5 (MatchText is itself a relevance filter).
6. `training/evaluate.py` — remove the unused `import numpy as np`; add `--limit` validation: after parse, `if args.limit is not None and args.limit <= 0: parser.error("--limit must be positive")`.
7. E741: rename the `l` loop variables in `api/services/transformer.py` (two dict comprehensions) to `label`; grep `for l ` / `for l,` across `api/ src/ training/ scripts/` and rename any others.
8. `tests/conftest.py` — FakeEmbedder docstring: "Deterministic within a process (hash-seeded)". `tests/unit/test_transformer_service.py::test_predict_is_deterministic` — add `explain=False` to both predict calls (drops pointless SHAP work).

- [ ] **Step 3: Run the suite**

Run: `pytest`
Expected: 104 passed (101 + 3 new).

- [ ] **Step 4: Commit**

```bash
git add -A -- api/ training/ tests/
git commit -m "refactor: retire deferred-minors backlog (guards, validation, renames)"
```

---

### Task 2: GitHub Actions CI + ruff

**Files:**
- Create: `.github/workflows/ci.yml`, `ruff.toml`
- Modify: `requirements-dev.txt` (+ `ruff>=0.4`)
- Test: local runs of ruff + pytest (CI itself verifies post-merge)

**Interfaces:** CI badge URL for Task 5: `https://github.com/evelindsayyy/lyrics_mood_predictor/actions/workflows/ci.yml/badge.svg`.

- [ ] **Step 1: ruff config and clean run**

`ruff.toml`:

```toml
line-length = 110
target-version = "py310"

[lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]  # long lines in docstrings/HTML blocks; length is advisory here

[lint.per-file-ignores]
"app/streamlit_app.py" = ["E402"]  # st.set_page_config ordering
"scripts/*.py" = ["E402"]          # sys.path bootstrap before imports
"training/*.py" = ["E402"]         # same
```

Append `ruff>=0.4` to `requirements-dev.txt`; `pip install -r requirements-dev.txt`.

Run: `ruff check .` — fix remaining violations the honest way: mechanical fixes (unused imports, import sorting via `ruff check --fix`) applied directly; anything non-mechanical gets a targeted per-file ignore WITH a comment. Document every violation you found and how you resolved it in your report. `ruff format` is NOT in scope (formatting churn).

- [ ] **Step 2: The workflow**

`.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install ruff
      - run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: requirements-dev.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: build api image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile.api
          push: false
      - name: build ui image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile.ui
          push: false
```

Validate YAML locally: `.venv/bin/python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('yaml ok')"` (pyyaml ships with jupyter deps; if missing, `pip install pyyaml`).

- [ ] **Step 3: Verify locally + commit**

Run: `ruff check .` → clean. `pytest` → 104 passed.

```bash
git add .github/workflows/ci.yml ruff.toml requirements-dev.txt <any lint-fixed files>
git commit -m "ci: add lint, test, and docker-build workflow"
```

---

### Task 3: Local-path Qdrant mode

**Files:**
- Modify: `api/config.py` (+ `qdrant_path: Path | None = None`), `api/services/retrieval.py` (`QdrantRetrieval.local(...)`), `api/main.py` (lifespan chooses local vs url), `scripts/index_corpus.py` (`--local-path`)
- Test: `tests/unit/test_local_qdrant.py`

**Interfaces:**
- `QdrantRetrieval.local(path: Path, collection: str = "songs") -> QdrantRetrieval` (classmethod; `QdrantClient(path=str(path))`; same search/find_song/ping/count surface).
- `Settings.qdrant_path: Path | None = None` — when set (env `LYRICMOOD_QDRANT_PATH`), lifespan builds `QdrantRetrieval.local(cfg.qdrant_path, cfg.qdrant_collection)` instead of the URL client.
- `python scripts/index_corpus.py --local-path demo/qdrant_local` — indexes into a file-based qdrant at that path instead of the URL. NOTE in script docs: local mode takes an exclusive lock — the API must not be running against the same path while indexing.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_local_qdrant.py`:

```python
"""Local-path (serverless) qdrant mode — index then serve from the same dir."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def local_dir(tmp_path):
    from qdrant_client import QdrantClient

    from scripts.index_corpus import ensure_collection, index_corpus

    df = pd.DataFrame(
        {
            "name": ["Midnight Rain", "Stadium Anthem"],
            "artist": ["Ava", "The Crowd"],
            "mood": ["Sad", "Hype"],
            "valence": [0.1, 0.9],
            "energy": [0.2, 0.9],
            "lyrics": ["rain empty street", "loud crowd jumping"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(2, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    d = tmp_path / "qdrant_local"
    client = QdrantClient(path=str(d))
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    client.close()
    return d, emb


def test_local_mode_serves_index(local_dir):
    from api.services.retrieval import QdrantRetrieval

    d, emb = local_dir
    r = QdrantRetrieval.local(d)
    assert r.ping() is True
    assert r.count() == 2
    hits = r.search(emb[0], limit=2)
    assert hits[0].title == "Midnight Rain"
    finds = r.find_song("stadium anthem")
    assert finds and finds[0].song_id == 1


def test_settings_qdrant_path(monkeypatch, tmp_path):
    from api.config import Settings

    monkeypatch.setenv("LYRICMOOD_QDRANT_PATH", str(tmp_path / "q"))
    assert str(Settings().qdrant_path) == str(tmp_path / "q")
    monkeypatch.delenv("LYRICMOOD_QDRANT_PATH")
    assert Settings().qdrant_path is None
```

Run → FAIL (`AttributeError: local`).

- [ ] **Step 2: Implement**

`api/services/retrieval.py` — add to `QdrantRetrieval`:

```python
    @classmethod
    def local(cls, path, collection: str = "songs") -> "QdrantRetrieval":
        """File-based serverless qdrant (demo deployments). Exclusive-lock:
        don't index and serve the same path concurrently."""
        self = cls.__new__(cls)
        self._client = QdrantClient(path=str(path))
        self._collection = collection
        return self
```

`api/config.py`: `qdrant_path: Path | None = None` (after `qdrant_url`).

`api/main.py` lifespan retrieval branch:

```python
        if not hasattr(app.state, "retrieval"):
            if cfg.qdrant_path is not None:
                app.state.retrieval = QdrantRetrieval.local(cfg.qdrant_path, cfg.qdrant_collection)
            else:
                app.state.retrieval = QdrantRetrieval(cfg.qdrant_url, cfg.qdrant_collection)
```

`scripts/index_corpus.py` — add argparse with `--local-path` (default None): when set, `client = QdrantClient(path=args.local_path)` else the existing URL client; print which mode. Keep the module's existing behavior for no-args invocation.

- [ ] **Step 3: Run the suite + commit**

Run: `pytest` → 106 passed (104 + 2).

```bash
git add api/config.py api/services/retrieval.py api/main.py scripts/index_corpus.py tests/unit/test_local_qdrant.py
git commit -m "feat: add serverless local-path qdrant mode for single-container demo"
```

---

### Task 4: Spaces demo package (bundle builder + image + local verification)

**Files:**
- Create: `scripts/build_demo_bundle.py`, `docker/Dockerfile.spaces`, `docker/spaces_launcher.sh`
- Modify: `.gitignore` (+ `demo/`)
- Test: real local build + run (no pytest)

**Interfaces:**
- `python scripts/build_demo_bundle.py [--out demo]` — assembles `demo/` (gitignored): `demo/models/` (best_classifier.pkl, tfidf_vectorizer.pkl, registry.json, transformer/, embedder/) copied from `models/`; `demo/qdrant_local/` built by invoking the indexer's local mode against the full corpus (excerpt payloads only — full lyrics never enter the bundle). Refuses (exit 1, clear message) if any source artifact is missing. Prints bundle size.
- `docker/Dockerfile.spaces`: python:3.11-slim; installs requirements-api.txt + requirements-ui.txt; copies `src/ api/ app/ demo/ docker/spaces_launcher.sh`; ENV `LYRICMOOD_MODEL_DIR=demo/models`, `LYRICMOOD_REGISTRY_PATH=demo/models/registry.json`, `LYRICMOOD_EMBEDDER_DIR=demo/models/embedder`, `LYRICMOOD_QDRANT_PATH=demo/qdrant_local`, `LYRICMOOD_API_URL=http://localhost:8000`; EXPOSE 7860; CMD the launcher. (Check `api/config.py` field names → env names; `labeled_songs_path` stays default and absent → lyrics store degrades by design.)
- `docker/spaces_launcher.sh`:

```bash
#!/bin/sh
set -e
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

- [ ] **Step 1: Implement the three files** (bundle builder mirrors the interface above; use `shutil.copytree/copy2`, `subprocess`-free — import the indexer's functions directly with a local `QdrantClient(path=...)`; embeddings from the cached `.npy` via `src.recommend.embed_corpus` like the indexer's main).

- [ ] **Step 2: Build the bundle for real**

Run: `.venv/bin/python scripts/build_demo_bundle.py`
Expected: `demo/` ≈ 400–600MB (models ~160MB + qdrant vectors/payloads), prints size. Add `demo/` to `.gitignore`.

- [ ] **Step 3: Build and run the demo container for real**

```bash
docker build -f docker/Dockerfile.spaces -t lyricmood-demo .
docker run -d --name lyricmood-demo -p 7860:7860 -p 8000:8000 lyricmood-demo   # 8000 exposed for verification only
sleep 25
curl -s localhost:8000/health          # transformer default, qdrant_ok true
curl -s "localhost:8000/v1/search?q=rainy%20late%20night%20drive&limit=3"      # 3 results
curl -s "localhost:8000/v1/songs?title=love" | head -c 400                     # candidates work
curl -s -o /dev/null -w "%{http_code}\n" localhost:7860                        # 200 (streamlit up)
docker rm -f lyricmood-demo
```

Capture real outputs. Note: `/v1/songs` single-match analysis should 503 `lyrics_unavailable` on this image — verify with a title that single-matches if you can find one quickly, otherwise state the expectation.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_demo_bundle.py docker/Dockerfile.spaces docker/spaces_launcher.sh .gitignore
git commit -m "feat: add single-container hf spaces demo package"
```

---

### Task 5: Deploy runbook + final README rewrite

**Files:**
- Create: `docs/DEPLOY_SPACES.md`
- Rewrite: `README.md` (structure below; keep Evaluation/Research/Contributions content)
- Modify: `CLAUDE.md` (on disk), `ATTRIBUTION.md` (week-4 entry)

- [ ] **Step 1: docs/DEPLOY_SPACES.md** — user-facing runbook: 1) `pip install huggingface_hub` + `huggingface-cli login`; 2) create a Docker-SDK Space (free CPU basic) named e.g. `lyricmood`; 3) clone the Space repo; copy in: `docker/Dockerfile.spaces` → `Dockerfile`, `docker/spaces_launcher.sh`, `src/`, `api/`, `app/`, `requirements-api.txt`, `requirements-ui.txt`, `demo/` (after running `build_demo_bundle.py`); 4) `git lfs track "demo/**" "*.onnx" "*.pkl" "*.npy"` + commit + push; 5) Space builds (~10 min), UI at the Space URL; 6) paste the URL into README's demo badge line. Include the copyright note (no full lyrics in the bundle; song-analysis endpoint degraded by design) and the free-tier sleep-after-inactivity caveat.

- [ ] **Step 2: README rewrite** — new structure (preserve existing content blocks where noted):

1. Title + one-line pitch + badges: CI (`[![ci](https://github.com/evelindsayyy/lyrics_mood_predictor/actions/workflows/ci.yml/badge.svg)](...)`) + `**[Live demo →](PASTE_SPACE_URL_HERE)**` placeholder line.
2. "What it does" — updated: three query types + two models via one API, UI is a client.
3. **Architecture** — ASCII diagram: ui (:8501) → api (:8000) → qdrant; offline lane: notebooks/Colab → artifacts → registry; label the torch-free ONNX serving path and the parity-checked embedder. Link `docs/architecture.md` and the spec.
4. API table: endpoint | what | example curl (predict, search, similar, songs, health, metrics).
5. Quick start (keep the compose block from Week 3).
6. Evaluation (keep the whole existing section verbatim — comparison table etc.).
7. "From class project to production system" — 6–8 bullet timeline of the elevation (what changed each week, one line each), linking the spec/plans; note LLM-assisted relabeling as future work.
8. Keep: Video Links, Research Connections, Individual Contributions, More Documentation, Repo Structure (update the tree: api/, training/, scripts/, tests/, docker/).

- [ ] **Step 3: CLAUDE.md (on disk) + ATTRIBUTION.md** — add demo-bundle/spaces commands and the local-qdrant mode note; ATTRIBUTION week-4 entry in house style.

- [ ] **Step 4: Verify + commit**

Run: `pytest` → 106 passed; `ruff check .` → clean.

```bash
git add docs/DEPLOY_SPACES.md README.md ATTRIBUTION.md
git commit -m "docs: final readme rewrite, spaces deploy runbook"
```

---

## Post-Merge (controller)

Merge → push → verify the CI run actually goes green on GitHub (`gh run watch` or the Actions page) — the workflow's first real execution is part of this week's definition of done. Then hand the user the DEPLOY_SPACES.md runbook.

## Out of Scope

LLM-relabeling experiment (documented as future work), README demo-URL fill-in (user, post-deploy), video re-recording.
