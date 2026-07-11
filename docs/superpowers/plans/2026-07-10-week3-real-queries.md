# LyricMood Week 3 — Real Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All three real query types live — free-text mood search (`GET /v1/search`), paste-lyrics similar songs (`POST /v1/similar`), and song title/artist lookup (`GET /v1/songs`) — plus slowapi rate limiting, Prometheus `/metrics`, and the Streamlit UI rewired as a pure API client (with a `ui` service in Docker Compose).

**Architecture:** Query-time embeddings come from a one-time ONNX export of `all-MiniLM-L6-v2` (`scripts/export_minilm_onnx.py`, parity-verified against sentence-transformers) served by `api/services/embedder.py` — the API stays torch-free. `RetrievalClient` grows `search()` (vector + optional mood filter) and `find_song()` (full-text payload match, difflib-ranked). `api/services/songs.py` holds a `LyricsStore` (song_id → lyrics from the processed CSV) for the lookup pipeline. All new heavy state loads in lifespan with the established eager-injection/`hasattr` pattern; absent artifacts degrade to 503s, never crash `/predict`.

**Tech Stack:** onnxruntime + tokenizers (existing), qdrant-client (existing), slowapi, prometheus-client, httpx (UI client), streamlit (UI only).

**Spec:** `docs/superpowers/specs/2026-07-09-industrial-elevation-design.md` §3.1 (search/songs contracts), §3.3 (query-type mapping), §5, Week-3 row of §7.

## Global Constraints

- Serving stays torch-free: `requirements-api.txt` may gain ONLY `slowapi` and `prometheus-client`. torch is used exclusively by the local one-time export script.
- `random_state=42` / seeded RNGs for anything random; NO `datetime.now()` in business logic (perf_counter for latency timing is fine).
- Every new module docstring carries an AI-attribution block (depth-correct path to `ATTRIBUTION.md`).
- Error envelope `{"error": {"code", "message"}}` everywhere; raw lyrics NEVER logged (log lengths/counts only); routes sync `def`.
- Degraded modes per spec §5: `/predict` must keep working when Qdrant/embedder/lyrics-store are absent; search/similar/songs return 503 (`retrieval_unavailable` / `search_unavailable` / `lyrics_unavailable`).
- All 60 existing tests keep passing; tests stay artifact-free and network-free (tiny ONNX fixtures).
- The full-lyrics embedding convention is headers-stripped RAW text (`re.sub(r"\[[^\]]*\]", " ", text)`) — must match how the corpus was embedded.
- `/health` and `/metrics` are exempt from rate limiting.
- Prometheus label cardinality: use the route TEMPLATE (e.g. `/v1/predict`), never raw paths/queries.

## File Structure

```
scripts/export_minilm_onnx.py        # one-time local export + parity check (torch here only)
api/services/embedder.py             # Embedder protocol, OnnxEmbedder, load_embedder, strip_section_headers
api/services/retrieval.py            # MODIFY: SongHit, search(), find_song(), collection param
api/services/songs.py                # LyricsStore
api/routes/search.py                 # GET /v1/search + POST /v1/similar
api/routes/songs.py                  # GET /v1/songs
api/ratelimit.py                     # slowapi limiter factory + envelope-preserving 429 handler
api/metrics.py                       # prometheus counters/histogram + middleware + /metrics route
api/schemas.py                       # MODIFY: SongResult, SearchResponse, SimilarRequest/Response, SongsResponse
api/config.py                        # MODIFY: embedder_dir, rate_limit
api/deps.py                          # MODIFY: get_embedder, get_lyrics_store
api/main.py                          # MODIFY: lifespan wiring, new routers, limiter, metrics
app/streamlit_app.py                 # REWRITE: pure API client (design/CSS unchanged)
requirements-ui.txt                  # streamlit + httpx only
docker/Dockerfile.ui
docker-compose.yml                   # MODIFY: + ui service
tests/conftest.py                    # MODIFY: tiny_embedder_dir fixture, FakeEmbedder, FakeRetrieval hits
tests/unit/test_embedder.py
tests/unit/test_retrieval_search.py
tests/unit/test_lyrics_store.py
tests/api/test_search.py
tests/api/test_songs.py
tests/api/test_ratelimit.py
tests/api/test_metrics.py
```

---

### Task 1: ONNX embedder (export script + serving service)

**Files:**
- Create: `scripts/export_minilm_onnx.py`, `api/services/embedder.py`
- Modify: `api/config.py` (add `embedder_dir`), `tests/conftest.py` (add `build_tiny_embedder_onnx` + `tiny_embedder_dir` fixture + `FakeEmbedder`)
- Test: `tests/unit/test_embedder.py`

**Interfaces:**
- Consumes: `ArtifactError` (api/services/model.py); tokenizer-building pattern from the existing `tiny_onnx_dir` fixture.
- Produces:
  - `strip_section_headers(text) -> str` in `api/services/embedder.py` (`re.sub(r"\[[^\]]*\]", " ", text)` on str, `""` otherwise — the corpus embedding convention).
  - `Embedder` Protocol: `embed(texts: list[str]) -> np.ndarray` — (n, d) float32, L2-normalized rows.
  - `OnnxEmbedder(session, tokenizer, max_len: int = 256)` implementing it: tokenize batch (padded) → run ONNX (`input_ids`, `attention_mask` int64 → `token_embeddings` float32 [batch, seq, dim]) → masked mean-pool → L2 normalize (both denominators clipped at 1e-9).
  - `load_embedder(model_dir: Path) -> OnnxEmbedder` — expects `model.onnx` + `tokenizer.json`; `ArtifactError` naming the first missing file.
  - `Settings.embedder_dir: Path = Path("models/embedder")`.
  - conftest: `build_tiny_embedder_onnx(vocab_size, dim, out_path)` (Gather graph emitting [batch, seq, dim]), session fixture `tiny_embedder_dir`, and `FakeEmbedder` (`embed` returns deterministic seeded normalized vectors, `dim=8` default).
  - CLI `python scripts/export_minilm_onnx.py [--out models/embedder]`: exports the HF model inside `sentence-transformers/all-MiniLM-L6-v2` to ONNX (fp32, no quantization — the vectors must match the corpus embedding space), saves `tokenizer.json`, then PARITY-CHECKS against `sentence_transformers.SentenceTransformer.encode(normalize_embeddings=True)` on 5 sentences (max abs cosine deviation < 1e-3) and prints PASS/FAIL, exiting 1 on FAIL.

- [ ] **Step 1: conftest additions**

Append to `tests/conftest.py`:

```python
def build_tiny_embedder_onnx(vocab_size: int, dim: int, out_path):
    """Tiny embedding graph with the real embedder I/O contract:
    Gather(table, input_ids) -> token_embeddings [batch, seq, dim].
    attention_mask declared (unused) so the serving feed dict matches."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    table = rng.normal(scale=0.5, size=(vocab_size, dim)).astype(np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Gather", ["table", "input_ids"], ["token_embeddings"])],
        name="tiny_embedder",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        outputs=[
            helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, ["batch", "seq", dim])
        ],
        initializer=[numpy_helper.from_array(table, name="table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


@pytest.fixture(scope="session")
def tiny_embedder_dir(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    d = tmp_path_factory.mktemp("tiny_embedder")
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        [t for t, _ in TINY_SONGS], WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    )
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.enable_truncation(max_length=32)
    tok.save(str(d / "tokenizer.json"))
    build_tiny_embedder_onnx(vocab_size=tok.get_vocab_size(), dim=8, out_path=d / "model.onnx")
    return d


class FakeEmbedder:
    """Deterministic text->vector fake for route tests."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self._dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_embedder.py`:

```python
"""Tests for api.services.embedder against the tiny embedder fixture."""

import numpy as np
import pytest


def _load(tiny_embedder_dir):
    from api.services.embedder import load_embedder

    return load_embedder(tiny_embedder_dir)


def test_strip_section_headers():
    from api.services.embedder import strip_section_headers

    assert "[Chorus]" not in strip_section_headers("[Chorus] loud crowd")
    assert strip_section_headers(None) == ""


def test_embed_shape_and_normalization(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    out = e.embed(["stadium lights bass", "rain empty street coat chair"])
    assert out.shape == (2, 8)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_is_deterministic(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    a = e.embed(["tender heart kitchen door"])
    b = e.embed(["tender heart kitchen door"])
    assert np.allclose(a, b)


def test_padding_does_not_change_embedding(tiny_embedder_dir):
    # same text alone vs batched with a longer neighbor must embed identically
    e = _load(tiny_embedder_dir)
    alone = e.embed(["stadium lights"])[0]
    batched = e.embed(["stadium lights", "rain empty street coat chair alone tonight"])[0]
    assert np.allclose(alone, batched, atol=1e-5)


def test_empty_text_does_not_crash(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    out = e.embed([""])
    assert out.shape == (1, 8)
    assert not np.isnan(out).any()


def test_load_embedder_missing_file(tmp_path):
    from api.services.embedder import load_embedder
    from api.services.model import ArtifactError

    with pytest.raises(ArtifactError) as exc:
        load_embedder(tmp_path)
    assert "model.onnx" in str(exc.value)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.embedder'`

- [ ] **Step 4: Implement the service**

`api/services/embedder.py`:

```python
"""
Query-time text embeddings — ONNX MiniLM behind an Embedder protocol.

The corpus matrix in Qdrant was embedded with sentence-transformers
all-MiniLM-L6-v2 (L2-normalized). Query vectors must live in the SAME space,
so this module serves a parity-checked ONNX export of that exact model
(scripts/export_minilm_onnx.py) — masked mean-pooling + L2 normalization
reimplemented in numpy. Serving stays torch-free.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.3 — query embedding must match corpus embedding space).
See ../../ATTRIBUTION.md.
"""

import re
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from api.services.model import ArtifactError

REQUIRED_FILES = ("model.onnx", "tokenizer.json")
_EPS = 1e-9


def strip_section_headers(text) -> str:
    """The corpus embedding convention: raw text minus [Verse]-style headers."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[[^\]]*\]", " ", text)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OnnxEmbedder:
    def __init__(self, session, tokenizer, max_len: int = 256):
        self._session = session
        self._tokenizer = tokenizer
        self._max_len = max_len

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        max_len = min(self._max_len, max(max(len(e.ids) for e in encodings), 1))
        ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            n = min(len(enc.ids), max_len)
            ids[i, :n] = enc.ids[:n]
            mask[i, :n] = enc.attention_mask[:n]
        (token_emb,) = self._session.run(
            ["token_embeddings"], {"input_ids": ids, "attention_mask": mask}
        )
        token_emb = np.asarray(token_emb, dtype=np.float32)
        m = mask[:, :, None].astype(np.float32)
        pooled = (token_emb * m).sum(axis=1) / np.clip(m.sum(axis=1), _EPS, None)
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), _EPS, None)
        return (pooled / norms).astype(np.float32)


def load_embedder(model_dir: Path, max_len: int = 256) -> OnnxEmbedder:
    model_dir = Path(model_dir)
    for name in REQUIRED_FILES:
        if not (model_dir / name).exists():
            raise ArtifactError(f"embedder artifact missing: {model_dir / name}")
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return OnnxEmbedder(session, tokenizer, max_len=max_len)
```

`api/config.py` — add after `registry_path`:

```python
    embedder_dir: Path = Path("models/embedder")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: 6 PASSED. `pytest` → 66 passed.

- [ ] **Step 6: Write the export script and run it for real**

`scripts/export_minilm_onnx.py`:

```python
"""
One-time export: all-MiniLM-L6-v2 -> models/embedder/{model.onnx,tokenizer.json}.

fp32 (NOT quantized) — query vectors must match the corpus embedding space,
so we verify parity against sentence-transformers before declaring success.
Run locally (torch required): python scripts/export_minilm_onnx.py

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PARITY_SENTENCES = [
    "stadium lights and a roaring crowd",
    "rain on the empty street tonight",
    "your hand in mine by the kitchen door",
    "tea gone cold by the window",
    "say it again, say it to my face",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/embedder")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.config.return_dict = False

    dummy = tokenizer(["dummy text for export"], return_tensors="pt", padding=True)
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(out_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["token_embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "token_embeddings": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))

    # parity check vs sentence-transformers
    from sentence_transformers import SentenceTransformer

    from api.services.embedder import load_embedder

    ours = load_embedder(out_dir).embed(PARITY_SENTENCES)
    ref = SentenceTransformer("all-MiniLM-L6-v2").encode(
        PARITY_SENTENCES, normalize_embeddings=True, convert_to_numpy=True
    )
    cos = (ours * ref).sum(axis=1)
    worst = float(1.0 - cos.min())
    print(f"parity: worst cosine deviation = {worst:.2e}")
    if worst > 1e-3:
        print("PARITY FAIL — do not serve this export")
        return 1
    print(f"PASS — artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `.venv/bin/python scripts/export_minilm_onnx.py`
Expected: downloads the HF model if needed, writes `models/embedder/model.onnx` (~90MB) + `tokenizer.json`, prints `parity: worst cosine deviation = ...` well under 1e-3 and `PASS`. Paste the real output in your report. NOTE: torch.onnx.export may emit TracerWarnings — fine; a PARITY FAIL is not.

Also add `models/embedder/` to `.gitignore` (next to `models/transformer/`).

- [ ] **Step 7: Commit**

```bash
git add api/services/embedder.py api/config.py scripts/export_minilm_onnx.py tests/conftest.py tests/unit/test_embedder.py .gitignore
git commit -m "feat: add torch-free onnx query embedder with parity-checked export"
```

---

### Task 2: Retrieval search + song lookup

**Files:**
- Modify: `api/services/retrieval.py`, `api/main.py` (pass `cfg.qdrant_collection`), `tests/conftest.py` (extend `FakeRetrieval`)
- Test: `tests/unit/test_retrieval_search.py`

**Interfaces:**
- Consumes: qdrant-client (`:memory:` in tests), collection schema from `scripts/index_corpus.py` (payload: song_id, title, artist, mood, valence, energy, lyrics_excerpt; indexes on mood/title/artist).
- Produces:
  - `SongHit` frozen dataclass: `song_id: int`, `title: str`, `artist: str`, `mood: str`, `score: float`.
  - `RetrievalClient` Protocol gains: `search(vector, limit: int = 10, mood: str | None = None) -> list[SongHit]`; `find_song(title: str, artist: str | None = None, limit: int = 5) -> list[SongHit]`.
  - `QdrantRetrieval(url, collection: str = "songs")` implements both: `search` = vector query (+ `mood` keyword filter), score = cosine similarity; `find_song` = full-text `MatchText` filter on title (+ artist when given) via `scroll`, then ranked client-side with `difflib.SequenceMatcher` ratio on lowercased title (score = ratio). Both return `[]` on any qdrant exception EXCEPT connection-level errors, which re-raise so routes can 503 — simpler and acceptable: let ALL exceptions propagate; routes translate to 503 (`ping()` stays the graceful one).
  - conftest `FakeRetrieval(ok=True, hits: list | None = None, find_hits: list | None = None)` — `search`/`find_song` return the canned lists (default `[]`); raises `RuntimeError` if `ok=False` (so route tests can simulate qdrant-down for 503 paths). Keep `ping()`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_retrieval_search.py`:

```python
"""QdrantRetrieval.search/find_song against in-memory qdrant with real indexed points."""

import numpy as np
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def seeded():
    """3 songs indexed exactly like scripts/index_corpus.py does."""
    import pandas as pd

    from api.services.retrieval import QdrantRetrieval
    from scripts.index_corpus import ensure_collection, index_corpus

    df = pd.DataFrame(
        {
            "name": ["Midnight Rain", "Stadium Anthem", "Kitchen Door"],
            "artist": ["Ava", "The Crowd", "June"],
            "mood": ["Sad", "Hype", "Romantic"],
            "valence": [0.1, 0.9, 0.7],
            "energy": [0.2, 0.9, 0.4],
            "lyrics": ["rain empty street", "loud crowd jumping", "hand in mine kitchen door"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(3, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")

    r = QdrantRetrieval.__new__(QdrantRetrieval)  # bypass URL ctor, inject memory client
    r._client = client
    r._collection = "songs"
    return r, emb


def test_search_returns_ranked_hits(seeded):
    r, emb = seeded
    hits = r.search(emb[0], limit=3)
    assert len(hits) == 3
    assert hits[0].song_id == 0  # exact vector match ranks first
    assert hits[0].title == "Midnight Rain"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_mood_filter(seeded):
    r, emb = seeded
    hits = r.search(emb[0], limit=3, mood="Hype")
    assert [h.mood for h in hits] == ["Hype"]
    assert hits[0].song_id == 1


def test_search_limit(seeded):
    r, emb = seeded
    assert len(r.search(emb[0], limit=1)) == 1


def test_find_song_by_title(seeded):
    r, _ = seeded
    hits = r.find_song("midnight rain")
    assert hits and hits[0].song_id == 0
    assert hits[0].score > 0.9  # near-exact title match


def test_find_song_with_artist(seeded):
    r, _ = seeded
    hits = r.find_song("kitchen door", artist="June")
    assert hits and hits[0].song_id == 2


def test_find_song_no_match(seeded):
    r, _ = seeded
    assert r.find_song("zzzz qqqq nonexistent") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_retrieval_search.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (no `search` on QdrantRetrieval).

- [ ] **Step 3: Implement**

In `api/services/retrieval.py` (extend; keep `ping` and attribution block, update the docstring's "Week 1 only needs ping" note):

```python
from dataclasses import dataclass
from difflib import SequenceMatcher

from qdrant_client import QdrantClient
from qdrant_client import models as qm


@dataclass(frozen=True)
class SongHit:
    song_id: int
    title: str
    artist: str
    mood: str
    score: float


class RetrievalClient(Protocol):
    def ping(self) -> bool: ...

    def search(self, vector, limit: int = 10, mood: str | None = None) -> list[SongHit]: ...

    def find_song(self, title: str, artist: str | None = None, limit: int = 5) -> list[SongHit]: ...


def _hit_from_payload(payload: dict, score: float) -> SongHit:
    return SongHit(
        song_id=int(payload["song_id"]),
        title=str(payload["title"]),
        artist=str(payload["artist"]),
        mood=str(payload["mood"]),
        score=float(score),
    )


class QdrantRetrieval:
    def __init__(self, url: str, collection: str = "songs"):
        self._client = QdrantClient(url=url, timeout=5)
        self._collection = collection

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def search(self, vector, limit: int = 10, mood: str | None = None) -> list[SongHit]:
        flt = None
        if mood is not None:
            flt = qm.Filter(must=[qm.FieldCondition(key="mood", match=qm.MatchValue(value=mood))])
        res = self._client.query_points(
            self._collection, query=list(map(float, vector)), limit=limit,
            query_filter=flt, with_payload=True,
        )
        return [_hit_from_payload(p.payload, p.score) for p in res.points]

    def find_song(self, title: str, artist: str | None = None, limit: int = 5) -> list[SongHit]:
        must = [qm.FieldCondition(key="title", match=qm.MatchText(text=title))]
        if artist:
            must.append(qm.FieldCondition(key="artist", match=qm.MatchText(text=artist)))
        points, _ = self._client.scroll(
            self._collection, scroll_filter=qm.Filter(must=must),
            limit=max(limit * 4, 20), with_payload=True,
        )
        scored = [
            _hit_from_payload(
                p.payload,
                SequenceMatcher(None, title.lower(), str(p.payload["title"]).lower()).ratio(),
            )
            for p in points
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]
```

`api/main.py`: `QdrantRetrieval(cfg.qdrant_url)` → `QdrantRetrieval(cfg.qdrant_url, cfg.qdrant_collection)`.

conftest `FakeRetrieval` — replace with:

```python
class FakeRetrieval:
    """Canned retrieval client for route tests."""

    def __init__(self, ok=True, hits=None, find_hits=None):
        self._ok = ok
        self._hits = list(hits or [])
        self._find_hits = list(find_hits or [])

    def ping(self):
        return self._ok

    def search(self, vector, limit=10, mood=None):
        if not self._ok:
            raise RuntimeError("qdrant down")
        out = [h for h in self._hits if mood is None or h.mood == mood]
        return out[:limit]

    def find_song(self, title, artist=None, limit=5):
        if not self._ok:
            raise RuntimeError("qdrant down")
        return self._find_hits[:limit]
```

NOTE: in-memory qdrant ignores payload text indexes but `MatchText` still filters by full-text matching in local mode; if `test_find_song_no_match` shows local-mode MatchText behaving differently (e.g. raising because no text index), adapt by creating the index in the seeded fixture via `ensure_collection` (already done) and document what you observed. Keep asserted behavior identical.

- [ ] **Step 4: Run the suite**

Run: `pytest`
Expected: 72 passed (66 + 6).

- [ ] **Step 5: Commit**

```bash
git add api/services/retrieval.py api/main.py tests/conftest.py tests/unit/test_retrieval_search.py
git commit -m "feat: add vector search and song lookup to retrieval client"
```

---

### Task 3: Lyrics store

**Files:**
- Create: `api/services/songs.py`
- Test: `tests/unit/test_lyrics_store.py`

**Interfaces:**
- Consumes: `Settings.labeled_songs_path`.
- Produces: `LyricsStore(lyrics: list)` with `get(song_id: int) -> str | None` (None when out of range or the stored value isn't a str) and `__len__`; `LyricsStore.from_csv(path: Path) -> LyricsStore` (reads ONLY the `lyrics` column, keeps row order — row position == indexer song_id).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_lyrics_store.py`:

```python
"""Tests for the song_id -> lyrics store."""


def test_get_by_row_position():
    from api.services.songs import LyricsStore

    s = LyricsStore(["first song words", "second song words"])
    assert s.get(0) == "first song words"
    assert s.get(1) == "second song words"
    assert len(s) == 2


def test_out_of_range_returns_none():
    from api.services.songs import LyricsStore

    s = LyricsStore(["only one"])
    assert s.get(5) is None
    assert s.get(-1) is None


def test_non_string_returns_none():
    from api.services.songs import LyricsStore

    s = LyricsStore([float("nan")])
    assert s.get(0) is None


def test_from_csv(tmp_path):
    import pandas as pd

    from api.services.songs import LyricsStore

    p = tmp_path / "songs.csv"
    pd.DataFrame({"lyrics": ["a b c", "d e f"], "mood": ["Sad", "Hype"]}).to_csv(p, index=False)
    s = LyricsStore.from_csv(p)
    assert len(s) == 2
    assert s.get(1) == "d e f"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_lyrics_store.py -v`
Expected: `ModuleNotFoundError: No module named 'api.services.songs'`

- [ ] **Step 3: Implement**

`api/services/songs.py`:

```python
"""
song_id -> full lyrics store for the /v1/songs pipeline.

Full lyrics never enter Qdrant (copyright + size); the indexer's point IDs
are row positions in the processed corpus, so this store is just the lyrics
column held in memory, indexed by position.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.3 — lyrics fetched from the local processed store by song_id).
See ../../ATTRIBUTION.md.
"""

from pathlib import Path

import pandas as pd


class LyricsStore:
    def __init__(self, lyrics: list):
        self._lyrics = list(lyrics)

    def __len__(self) -> int:
        return len(self._lyrics)

    def get(self, song_id: int) -> str | None:
        if not 0 <= song_id < len(self._lyrics):
            return None
        value = self._lyrics[song_id]
        return value if isinstance(value, str) else None

    @classmethod
    def from_csv(cls, path: Path) -> "LyricsStore":
        df = pd.read_csv(path, usecols=["lyrics"])
        return cls(df["lyrics"].tolist())
```

- [ ] **Step 4: Run the suite**

Run: `pytest` → 76 passed.

- [ ] **Step 5: Commit**

```bash
git add api/services/songs.py tests/unit/test_lyrics_store.py
git commit -m "feat: add lyrics store keyed by indexer song_id"
```

---

### Task 4: Search + similar routes

**Files:**
- Create: `api/routes/search.py`
- Modify: `api/schemas.py` (new response models), `api/deps.py` (get_embedder), `api/main.py` (lifespan + router + create_app params)
- Test: `tests/api/test_search.py`

**Interfaces:**
- Consumes: `Embedder`/`load_embedder`/`strip_section_headers` (Task 1), `RetrievalClient.search`/`SongHit`/`FakeRetrieval`/`FakeEmbedder` (Tasks 1-2).
- Produces:
  - Schemas: `SongResult(song_id: int, title: str, artist: str, mood: str, score: float)`; `SearchResponse(results: list[SongResult], query_embedding_ms: float, total_ms: float)`; `SimilarRequest(lyrics: str = Field(max_length=10_000), mood: str | None = None, limit: int = Field(5, ge=1, le=20))`.
  - `GET /v1/search?q=<3..200 chars>&limit=<1..20, default 10>&mood=<optional>`: q length enforced via `Query(min_length=3, max_length=200)` (422 outside); mood not one of the 5 → 400 `invalid_mood`; embedder absent → 503 `search_unavailable`; retrieval raising → 503 `retrieval_unavailable`. Flow: embed q → `retrieval.search` → `SearchResponse`. Never log q itself (log its length).
  - `POST /v1/similar` body `SimilarRequest`: empty/whitespace lyrics → 400 `empty_lyrics`; embeds `strip_section_headers(lyrics)`; same mood validation + 503s; returns `SearchResponse` (query_embedding_ms + total_ms populated).
  - `api/deps.py`: `get_embedder(request) -> Embedder | None` and `get_lyrics_store(request)` via `getattr(app.state, ..., None)`.
  - `create_app(..., embedder=None, lyrics_store=None)`: provided (non-None) values set eagerly; lifespan `hasattr` guard loads real ones — `load_embedder(cfg.embedder_dir)` if the dir exists else `None` (info log `embedder_unavailable`), `LyricsStore.from_csv(cfg.labeled_songs_path)` if the file exists else `None` (info log `lyrics_unavailable`).
  - `MOODS = {"Hype", "Romantic", "Calm", "Sad", "Angry"}` constant in `api/routes/search.py` (import into songs route later).

- [ ] **Step 1: Write the failing tests**

`tests/api/test_search.py`:

```python
"""Contract tests for GET /v1/search and POST /v1/similar."""

from fastapi.testclient import TestClient

from api.services.retrieval import SongHit
from tests.conftest import FakeEmbedder, FakeMoodModel, FakeRetrieval

HITS = [
    SongHit(song_id=1, title="Stadium Anthem", artist="The Crowd", mood="Hype", score=0.91),
    SongHit(song_id=2, title="Midnight Rain", artist="Ava", mood="Sad", score=0.80),
]


def _client(retrieval=None, embedder=FakeEmbedder()):
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()},
        default="baseline",
        retrieval=retrieval or FakeRetrieval(hits=HITS),
        embedder=embedder,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_search_happy_path():
    r = _client().get("/v1/search?q=rainy late night drive")
    assert r.status_code == 200
    body = r.json()
    assert [x["title"] for x in body["results"]] == ["Stadium Anthem", "Midnight Rain"]
    assert body["query_embedding_ms"] >= 0
    assert body["total_ms"] >= body["query_embedding_ms"]


def test_search_mood_filter():
    r = _client().get("/v1/search?q=rainy late night drive&mood=Sad")
    assert [x["mood"] for x in r.json()["results"]] == ["Sad"]


def test_search_invalid_mood_400():
    r = _client().get("/v1/search?q=rainy drive&mood=Moody")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_mood"


def test_search_query_too_short_422():
    r = _client().get("/v1/search?q=ab")
    assert r.status_code == 422


def test_search_embedder_absent_503():
    from api.main import create_app

    app = create_app(
        models={"baseline": FakeMoodModel()}, default="baseline", retrieval=FakeRetrieval(hits=HITS)
    )
    app.state.embedder = None
    r = TestClient(app, raise_server_exceptions=False).get("/v1/search?q=rainy drive")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "search_unavailable"


def test_search_retrieval_down_503():
    r = _client(retrieval=FakeRetrieval(ok=False)).get("/v1/search?q=rainy drive")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "retrieval_unavailable"


def test_similar_happy_path():
    r = _client().post("/v1/similar", json={"lyrics": "[Chorus] rain on the street " * 20})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_similar_empty_lyrics_400():
    r = _client().post("/v1/similar", json={"lyrics": "   "})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_lyrics"


def test_similar_mood_filter_and_limit():
    r = _client().post("/v1/similar", json={"lyrics": "rain street", "mood": "Hype", "limit": 1})
    body = r.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["mood"] == "Hype"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/api/test_search.py -v`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'embedder'`.

- [ ] **Step 3: Implement**

`api/schemas.py` — append:

```python
class SongResult(BaseModel):
    song_id: int
    title: str
    artist: str
    mood: str
    score: float


class SearchResponse(BaseModel):
    results: list[SongResult]
    query_embedding_ms: float
    total_ms: float


class SimilarRequest(BaseModel):
    lyrics: str = Field(max_length=10_000)
    mood: str | None = None
    limit: int = Field(5, ge=1, le=20)
```

`api/deps.py` — append:

```python
def get_embedder(request: Request):
    return getattr(request.app.state, "embedder", None)


def get_lyrics_store(request: Request):
    return getattr(request.app.state, "lyrics_store", None)
```

`api/routes/search.py`:

```python
"""
GET /v1/search — free-text mood search; POST /v1/similar — paste-lyrics
similar songs. Both: embed with the ONNX MiniLM -> vector search in Qdrant.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 search contract, §3.3 query mapping). See ../../ATTRIBUTION.md.
"""

import time

import structlog
from fastapi import APIRouter, Depends, Query

from api.deps import get_embedder, get_retrieval
from api.errors import ApiError
from api.schemas import SearchResponse, SimilarRequest, SongResult
from api.services.embedder import strip_section_headers

router = APIRouter()
logger = structlog.get_logger()

MOODS = {"Hype", "Romantic", "Calm", "Sad", "Angry"}


def _validate_mood(mood: str | None) -> None:
    if mood is not None and mood not in MOODS:
        raise ApiError(400, "invalid_mood", f"mood must be one of {sorted(MOODS)}")


def _run_query(text: str, mood: str | None, limit: int, embedder, retrieval) -> SearchResponse:
    if embedder is None:
        raise ApiError(503, "search_unavailable", "query embedder not loaded")
    t0 = time.perf_counter()
    vector = embedder.embed([text])[0]
    t_embed = time.perf_counter()
    try:
        hits = retrieval.search(vector, limit=limit, mood=mood)
    except Exception as exc:
        logger.warning("retrieval_error", error=type(exc).__name__)
        raise ApiError(503, "retrieval_unavailable", "vector search is unavailable") from exc
    t1 = time.perf_counter()
    return SearchResponse(
        results=[SongResult(**h.__dict__) for h in hits],
        query_embedding_ms=(t_embed - t0) * 1000,
        total_ms=(t1 - t0) * 1000,
    )


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=3, max_length=200),
    limit: int = Query(10, ge=1, le=20),
    mood: str | None = Query(None),
    embedder=Depends(get_embedder),
    retrieval=Depends(get_retrieval),
) -> SearchResponse:
    _validate_mood(mood)
    response = _run_query(q, mood, limit, embedder, retrieval)
    logger.info("search", query_chars=len(q), n_results=len(response.results))
    return response


@router.post("/similar", response_model=SearchResponse)
def similar(
    req: SimilarRequest,
    embedder=Depends(get_embedder),
    retrieval=Depends(get_retrieval),
) -> SearchResponse:
    text = strip_section_headers(req.lyrics).strip()
    if not text:
        raise ApiError(400, "empty_lyrics", "lyrics must contain non-whitespace text")
    _validate_mood(req.mood)
    response = _run_query(text, req.mood, req.limit, embedder, retrieval)
    logger.info("similar", input_chars=len(text), n_results=len(response.results))
    return response
```

`api/main.py` — `create_app` gains `embedder=None, lyrics_store=None` params; eager block sets them when not None; lifespan gains (after retrieval):

```python
        if not hasattr(app.state, "embedder"):
            if cfg.embedder_dir.exists():
                app.state.embedder = load_embedder(cfg.embedder_dir)
            else:
                logger.info("embedder_unavailable", dir=str(cfg.embedder_dir))
                app.state.embedder = None
        if not hasattr(app.state, "lyrics_store"):
            if cfg.labeled_songs_path.exists():
                app.state.lyrics_store = LyricsStore.from_csv(cfg.labeled_songs_path)
            else:
                logger.info("lyrics_unavailable", path=str(cfg.labeled_songs_path))
                app.state.lyrics_store = None
```

plus `app.include_router(search.router, prefix="/v1")` and imports (`load_embedder`, `LyricsStore`, `search` route module).

- [ ] **Step 4: Run the suite**

Run: `pytest` → 86 passed (76 + 10).

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py api/deps.py api/routes/search.py api/main.py tests/api/test_search.py
git commit -m "feat: add free-text search and paste-lyrics similar endpoints"
```

---

### Task 5: Song lookup route

**Files:**
- Create: `api/routes/songs.py`
- Modify: `api/schemas.py` (SongsResponse), `api/main.py` (router)
- Test: `tests/api/test_songs.py`

**Interfaces:**
- Consumes: `RetrievalClient.find_song`/`search`, `LyricsStore.get`, `MOODS` (Task 4), default `MoodModel`, `Embedder`.
- Produces:
  - Schema `SongAnalysis(mood: str, confidence: float, probabilities: dict[str, float], model_version: str)`; `SongsResponse(match: SongResult | None, analysis: SongAnalysis | None, similar: list[SongResult], candidates: list[SongResult])`.
  - `GET /v1/songs?title=<2..200 required>&artist=<optional>`: retrieval raising → 503 `retrieval_unavailable`; 0 hits → 404 `song_not_found`; >1 hits → `match=None, analysis=None, similar=[], candidates=<ranked hits>`; exactly 1 hit → full pipeline: lyrics = `lyrics_store.get(song_id)` (store None → 503 `lyrics_unavailable`; song missing from store → 404 `song_not_found`) → default model predict (`explain=False`) → similar = embed stripped lyrics → `retrieval.search(mood=predicted, limit=6)` minus the matched `song_id`, first 5 (embedder None → `similar=[]`, response still 200 — lookup degrades gracefully, search does not block analysis).
  - Song titles/artists are fine to log (public metadata, not lyrics).

- [ ] **Step 1: Write the failing tests**

`tests/api/test_songs.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/api/test_songs.py -v`
Expected: FAIL (404s from missing route).

- [ ] **Step 3: Implement**

`api/schemas.py` — append:

```python
class SongAnalysis(BaseModel):
    mood: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class SongsResponse(BaseModel):
    match: SongResult | None
    analysis: SongAnalysis | None
    similar: list[SongResult]
    candidates: list[SongResult]
```

`api/routes/songs.py`:

```python
"""
GET /v1/songs — look a song up by title/artist, run the full mood pipeline
on its lyrics, and return 5 similar songs. Multiple fuzzy matches return
ranked candidates instead. Titles/artists are public metadata and may be
logged; lyrics still never are.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 songs contract, §4 request flow). See ../../ATTRIBUTION.md.
"""

import structlog
from fastapi import APIRouter, Depends, Query, Request

from api.deps import get_default_model_name, get_embedder, get_lyrics_store, get_models, get_retrieval
from api.errors import ApiError
from api.schemas import SongAnalysis, SongResult, SongsResponse
from api.services.embedder import strip_section_headers

router = APIRouter()
logger = structlog.get_logger()


@router.get("/songs", response_model=SongsResponse)
def songs(
    request: Request,
    title: str = Query(min_length=2, max_length=200),
    artist: str | None = Query(None, max_length=200),
    retrieval=Depends(get_retrieval),
    embedder=Depends(get_embedder),
    lyrics_store=Depends(get_lyrics_store),
    models=Depends(get_models),
    default_name: str = Depends(get_default_model_name),
) -> SongsResponse:
    try:
        hits = retrieval.find_song(title, artist=artist)
    except Exception as exc:
        logger.warning("retrieval_error", error=type(exc).__name__)
        raise ApiError(503, "retrieval_unavailable", "song lookup is unavailable") from exc

    if not hits:
        raise ApiError(404, "song_not_found", f"no song matching title {title!r}")

    to_result = lambda h: SongResult(**h.__dict__)  # noqa: E731

    if len(hits) > 1:
        logger.info("songs_candidates", title=title, n=len(hits))
        return SongsResponse(match=None, analysis=None, similar=[], candidates=[to_result(h) for h in hits])

    hit = hits[0]
    if lyrics_store is None:
        raise ApiError(503, "lyrics_unavailable", "lyrics store not loaded")
    lyrics = lyrics_store.get(hit.song_id)
    if lyrics is None:
        raise ApiError(404, "song_not_found", f"lyrics missing for song_id {hit.song_id}")

    model = models[default_name]
    result = model.predict(lyrics, explain=False)
    analysis = SongAnalysis(
        mood=result.mood,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_version=model.version,
    )

    similar: list[SongResult] = []
    if embedder is not None:
        vector = embedder.embed([strip_section_headers(lyrics)])[0]
        try:
            raw = retrieval.search(vector, limit=6, mood=result.mood)
            similar = [to_result(h) for h in raw if h.song_id != hit.song_id][:5]
        except Exception as exc:
            logger.warning("retrieval_error", error=type(exc).__name__)

    logger.info("songs_match", title=hit.title, song_id=hit.song_id, mood=result.mood)
    return SongsResponse(match=to_result(hit), analysis=analysis, similar=similar, candidates=[])
```

`api/main.py`: `app.include_router(songs.router, prefix="/v1")` + import.

- [ ] **Step 4: Run the suite**

Run: `pytest` → 93 passed (86 + 7).

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py api/routes/songs.py api/main.py tests/api/test_songs.py
git commit -m "feat: add song title/artist lookup with full mood analysis"
```

---

### Task 6: Rate limiting (slowapi)

**Files:**
- Create: `api/ratelimit.py`
- Modify: `requirements-api.txt` (+ `slowapi>=0.1.9`), `api/config.py` (+ `rate_limit: str = "30/minute"`), `api/main.py` (wire limiter)
- Test: `tests/api/test_ratelimit.py`

**Interfaces:**
- Produces: `build_limiter(rate_limit: str) -> Limiter` (key_func = client IP, `default_limits=[rate_limit]`); `rate_limit_handler(request, exc) -> JSONResponse` returning the envelope `{"error": {"code": "rate_limited", "message": ...}}` with status 429 and a `Retry-After` header. `create_app` builds the limiter from `cfg.rate_limit`, sets `app.state.limiter`, registers `SlowAPIMiddleware` + the handler, and EXEMPTS `/health` and `/metrics` (via `limiter.exempt(...)` on those endpoint functions, or the installed slowapi version's equivalent — if the exempt API differs, achieve the same effect and document how in your report).

- [ ] **Step 1: Add dependency**

Append `slowapi>=0.1.9` to `requirements-api.txt`; `pip install -r requirements-dev.txt`.

- [ ] **Step 2: Write the failing tests**

`tests/api/test_ratelimit.py`:

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/api/test_ratelimit.py -v`
Expected: FAIL — no 429 (limit not enforced yet) / TypeError on Settings field.

- [ ] **Step 4: Implement**

`api/config.py` — add `rate_limit: str = "30/minute"`.

`api/ratelimit.py`:

```python
"""
slowapi rate limiting with the project's error envelope.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 — 30 req/min/IP, 429 + Retry-After). See ../ATTRIBUTION.md.
"""

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def build_limiter(rate_limit: str) -> Limiter:
    return Limiter(key_func=get_remote_address, default_limits=[rate_limit])


def rate_limit_handler(request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={"error": {"code": "rate_limited", "message": f"rate limit exceeded: {exc.detail}"}},
    )
    response.headers["Retry-After"] = "60"
    return response
```

`api/main.py` — in `create_app`, after `register_exception_handlers(app)`:

```python
    limiter = build_limiter(cfg.rate_limit)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)
    limiter.exempt(health.health)
```

with imports `from slowapi.errors import RateLimitExceeded`, `from slowapi.middleware import SlowAPIMiddleware`, `from api.ratelimit import build_limiter, rate_limit_handler`. NOTE: each `create_app` call builds a FRESH Limiter (fresh in-memory counters) so tests don't bleed into each other — verify this holds; if slowapi keeps global state across instances, isolate per-test (e.g. unique limiter storage) and document it.

- [ ] **Step 5: Run the suite**

Run: `pytest` → 96 passed (93 + 3).

- [ ] **Step 6: Commit**

```bash
git add requirements-api.txt api/config.py api/ratelimit.py api/main.py tests/api/test_ratelimit.py
git commit -m "feat: add per-ip rate limiting with envelope 429s"
```

---

### Task 7: Prometheus /metrics

**Files:**
- Create: `api/metrics.py`
- Modify: `requirements-api.txt` (+ `prometheus-client>=0.20`), `api/main.py` (middleware + route)
- Test: `tests/api/test_metrics.py`

**Interfaces:**
- Produces: `api/metrics.py` with `REQUEST_COUNT = Counter("lyricmood_requests_total", ..., ["path", "method", "status"])`, `REQUEST_LATENCY = Histogram("lyricmood_request_seconds", ..., ["path"])`, `metrics_middleware(request, call_next)` (labels use the matched route TEMPLATE via `request.scope.get("route").path` when available, else `"unmatched"`), and `metrics_endpoint()` returning `generate_latest()` with `CONTENT_TYPE_LATEST`. Mounted as `GET /metrics` (no /v1 prefix, rate-limit exempt).

- [ ] **Step 1: Write the failing tests**

`tests/api/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/api/test_metrics.py -v`
Expected: 404 on /metrics.

- [ ] **Step 3: Implement**

Append `prometheus-client>=0.20` to `requirements-api.txt`; install.

`api/metrics.py`:

```python
"""
Prometheus metrics: request counts + latency histograms per route template.

Route TEMPLATES (e.g. /v1/predict) keep label cardinality bounded — raw
paths/queries never become label values.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 metrics endpoint). See ../ATTRIBUTION.md.
"""

import time

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "lyricmood_requests_total", "HTTP requests", ["path", "method", "status"]
)
REQUEST_LATENCY = Histogram("lyricmood_request_seconds", "Request latency", ["path"])


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    if path not in ("/metrics",):
        REQUEST_COUNT.labels(path=path, method=request.method, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(path=path).observe(time.perf_counter() - start)
    return response


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

`api/main.py`: `app.middleware("http")(metrics_middleware)` (register BEFORE `request_id_middleware` registration or after — order is not contractual; pick one and note it), `app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])`, and exempt it from rate limiting alongside health (`limiter.exempt(metrics_endpoint)` — same caveat as Task 6).

NOTE: `request.scope.get("route")` is set by the router only for matched routes; for a 404 it stays unset → `"unmatched"` label. Middleware runs for exception-handled responses because handlers produce responses inside `call_next`'s downstream — EXCEPT the request-id middleware's own 500 catch; that path yields a response normally through the stack, so it is counted. Verify with the tests; if the counting for 400s doesn't appear, check middleware registration order and document the final order.

- [ ] **Step 4: Run the suite**

Run: `pytest` → 99 passed (96 + 3).

- [ ] **Step 5: Commit**

```bash
git add requirements-api.txt api/metrics.py api/main.py tests/api/test_metrics.py
git commit -m "feat: add prometheus metrics endpoint and request instrumentation"
```

---

### Task 8: Streamlit as pure API client (+ ui container)

**Files:**
- Rewrite: `app/streamlit_app.py` (keep design/CSS/copy identical; swap the backend)
- Create: `requirements-ui.txt`, `docker/Dockerfile.ui`
- Modify: `docker-compose.yml` (+ ui service)
- Test: manual smoke (this task has no pytest — the UI is display glue; the API contracts it consumes are already covered)

**Interfaces:**
- Consumes: `POST /v1/predict` (mood/confidence/probabilities/explanation/model_version/warnings), `POST /v1/similar` (results with title/artist/score).
- Produces: a Streamlit app whose ONLY backend dependency is HTTP: `LYRICMOOD_API_URL` env (default `http://localhost:8000`). No joblib/sklearn/shap/sentence-transformers/pandas imports remain in `app/`.

- [ ] **Step 1: Rewrite app/streamlit_app.py**

KEEP VERBATIM (do not touch): the module sections `FONTS_LINK`, `_load_css`, `STREAMLIT_OVERRIDES`, `inject_design_system`, `set_mood_accent`, `st.set_page_config(...)` + `inject_design_system()` calls, `MOOD_ORDER`, `MOOD_BLURB`, `SAMPLES`, the session-state block, `set_sample`/`clear_all`, the brand/prompt markdown, the textarea/chipbar/action-row blocks, and ALL the result-rendering markdown blocks (reading header, probability stack, SHAP chart HTML, similar-songs list).

REPLACE the module docstring (update it to describe the API-client architecture, keep an AI-attribution block), REPLACE the imports + loader + pipeline sections as follows:

Imports become:

```python
import os

import httpx
import streamlit as st
```

(delete: ast, re, sys, joblib, numpy, pandas, src.* imports and the sys.path insert.)

The cached loader becomes:

```python
API_URL = os.environ.get("LYRICMOOD_API_URL", "http://localhost:8000")


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30.0)


def _api_error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return f"API error (HTTP {response.status_code})"
```

The `if go_clicked:` pipeline block becomes:

```python
if go_clicked:
    text = (st.session_state["lyrics"] or "").strip()
    if not text:
        st.warning("paste some lyrics first 🙂")
        st.stop()

    with st.spinner("reading the room…"):
        client = api_client()
        try:
            pred_resp = client.post("/v1/predict", json={"lyrics": text})
        except httpx.HTTPError:
            st.error(f"can't reach the LyricMood API at {API_URL} — is it running? (docker compose up)")
            st.stop()
        if pred_resp.status_code != 200:
            st.error(_api_error_message(pred_resp))
            st.stop()
        pred = pred_resp.json()

        top10 = [(e["token"], e["weight"]) for e in (pred["explanation"] or [])]

        recs = []
        sim_resp = client.post(
            "/v1/similar", json={"lyrics": text, "mood": pred["mood"], "limit": 5}
        )
        if sim_resp.status_code == 200:
            recs = [
                {"title": r["title"], "artist": r["artist"], "similarity": r["score"]}
                for r in sim_resp.json()["results"]
            ]

        st.session_state["result"] = {
            "pred": pred["mood"],
            "confidence": pred["confidence"],
            "prob_map": pred["probabilities"],
            "top10": top10,
            "recs": recs,
        }
```

The rendering block consumes `st.session_state["result"]` with the SAME keys as before, so it stays verbatim — with ONE guard added around the similar-songs section: if `recs` is empty, render the section with a single muted row `retrieval offline — similar songs unavailable` instead of the list (use the existing `.lab` styling; a plain `<div class="lab" style="padding: 18px;">` inside `.list` is fine).

One more addition: `prob_map` from the transformer may not include all 5 moods at probability>0 rendering fine as-is (`prob_map.get(m, 0.0)` already guards) — verify that guard survived the rewrite untouched.

- [ ] **Step 2: UI packaging**

`requirements-ui.txt`:

```
streamlit>=1.30
httpx>=0.27
```

`docker/Dockerfile.ui`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY app/ app/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

`docker-compose.yml` — add:

```yaml
  ui:
    build:
      context: .
      dockerfile: docker/Dockerfile.ui
    ports:
      - "8501:8501"
    environment:
      LYRICMOOD_API_URL: http://api:8000
    depends_on:
      - api
```

- [ ] **Step 3: Manual smoke (real stack)**

Run: `docker compose up -d --build && sleep 25`
Then:
- `curl -s localhost:8000/health` → transformer default, `qdrant_ok: true`.
- Open `http://localhost:8501`, paste the Sad sample, click Read the mood → mood + confidence + SHAP tokens + 5 similar songs render.
- `curl -s "localhost:8000/v1/search?q=rainy%20late%20night%20drive&limit=3"` → 3 results with scores.
- `curl -s "localhost:8000/v1/songs?title=love"` → candidates or a match.
- `curl -s localhost:8000/metrics | head -5` → prometheus text.
- Fire 35 rapid requests: `for i in $(seq 35); do curl -s -o /dev/null -w "%{http_code} " localhost:8000/v1/predict -X POST -H 'content-type: application/json' -d '{"lyrics":"test"}'; done` → 200s then 429s.

Capture real outputs in your report. `docker compose down` after.

NOTE: the api image build must now include `models/embedder/` at runtime — it arrives via the existing `./models:/app/models:ro` volume mount (created in Task 1's export), NOT baked into the image. Verify `/v1/search` works in-container; if it 503s, check the mount.

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py requirements-ui.txt docker/Dockerfile.ui docker-compose.yml
git commit -m "feat: rewire streamlit as pure api client with ui container"
```

---

### Task 9: Docs

**Files:**
- Modify: `README.md` (Quick Start: three-service compose + new endpoints), `CLAUDE.md` (on-disk only: commands + architecture), `ATTRIBUTION.md` (week-3 modules), `SETUP.md` (step for `scripts/export_minilm_onnx.py`)

**Steps:**

- [ ] **Step 1:** README Quick Start — replace the week-1 API block with:

```markdown
# 4. run the full stack (api + vector db + web ui)
docker compose up --build          # ui :8501, api :8000, qdrant :6333
python scripts/index_corpus.py     # one-time corpus indexing
python scripts/export_minilm_onnx.py  # one-time query-embedder export

# then: open http://localhost:8501, or hit the API directly —
curl -X POST localhost:8000/v1/predict -H 'content-type: application/json' -d '{"lyrics": "..."}'
curl "localhost:8000/v1/search?q=rainy%20late%20night%20drive"
curl "localhost:8000/v1/songs?title=midnight"
```

- [ ] **Step 2:** CLAUDE.md (on disk, untracked): add the two one-time scripts + the three query endpoints to Commands; extend the API architecture paragraph with: embedder (`api/services/embedder.py`, ONNX MiniLM, parity-checked, `models/embedder/` local-only), retrieval search/find_song, LyricsStore, slowapi rate limiting (`rate_limit` setting), prometheus `/metrics`, and "app/streamlit_app.py is a pure API client — no model imports; `LYRICMOOD_API_URL` env".

- [ ] **Step 3:** ATTRIBUTION.md — extend the api/ entry with the week-3 modules (embedder, retrieval search, songs, ratelimit, metrics, streamlit rewrite) in the established style. SETUP.md — add the embedder-export one-liner after the corpus-embedding step.

- [ ] **Step 4:** `pytest` → 99 passed. Commit:

```bash
git add README.md ATTRIBUTION.md SETUP.md
git commit -m "docs: document week-3 query endpoints, ui service, and embedder export"
```

---

## Out of Scope (Week 4)

CI (GitHub Actions), HF Spaces deploy, README final rewrite/architecture diagram, LLM-relabeling stretch, deferred minors in `.superpowers/sdd/progress.md`.
