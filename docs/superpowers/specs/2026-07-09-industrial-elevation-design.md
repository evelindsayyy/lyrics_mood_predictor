# LyricMood Industrial Elevation — Design Spec

**Date:** 2026-07-09
**Status:** Approved design, pending implementation plan
**Approach:** A — service-first re-platform with the model upgrade riding inside it

## 1. Context & Goals

LyricMood is currently a class project: Streamlit app + notebooks + pickled sklearn artifacts. The owner wants to elevate it into an industrial-grade portfolio piece.

**Decisions from requirements exploration:**

| question | decision |
|---|---|
| Purpose | Portfolio / job-hunting credibility |
| Target role | AI Engineer (LLM-era) |
| Budget | Free tier only (Colab free GPU, free hosting, no paid APIs) |
| Timeline | ~1 month part-time |
| "Real queries" means | (a) concurrent API traffic, (b) natural-language mood search, (c) song title/artist lookup |

**Goals:**

1. A production-grade REST API replacing Streamlit-as-backend, handling concurrent traffic with validation, rate limiting, and observability.
2. Measurably better accuracy: fine-tuned transformer benchmarked against the existing TF-IDF+LR baseline (test accuracy 0.432 / macro F1 0.371).
3. Three real query types: paste-lyrics predict, free-text semantic mood search, song title/artist lookup.
4. Industrial stack legible to interviewers: FastAPI, Qdrant, ONNX, MLflow, Docker Compose, GitHub Actions CI, HF Spaces demo.

**Non-goals (explicitly out of scope for the month):**

- Fixing label noise (valence/energy thresholding stays as-is; LLM-relabeling is a stretch goal).
- Catalog ingestion / growing beyond the 76k SpotGenTrack corpus.
- Kubernetes, managed cloud infra, paid hosting, auth/user accounts.
- Replacing the Streamlit UI's visual design (it stays, demoted to an API client).

## 2. Architecture

```
                        ┌─────────────────────────────────────────┐
                        │        clients (Streamlit UI, curl,     │
                        │        anyone hitting the REST API)     │
                        └───────────────────┬─────────────────────┘
                                            │ HTTP/JSON
                        ┌───────────────────▼─────────────────────┐
                        │   FastAPI service  (api/)                │
                        │   POST /v1/predict   lyrics → mood       │
                        │   GET  /v1/search    free text → songs   │
                        │   GET  /v1/songs     title/artist lookup │
                        │   GET  /health, /metrics                 │
                        └──────┬──────────────────────┬────────────┘
                               │                      │
                 ┌─────────────▼──────────┐   ┌───────▼───────────────┐
                 │  Model service layer   │   │  Qdrant (Docker)      │
                 │  - fine-tuned          │   │  76k songs: MiniLM    │
                 │    transformer (ONNX,  │   │  vectors + payload    │
                 │    CPU inference)      │   │  {title, artist,      │
                 │  - TF-IDF+LR kept as   │   │   mood, excerpt}      │
                 │    fast/explainable    │   │  similar-songs, NL    │
                 │    fallback            │   │  search, title lookup │
                 └─────────────▲──────────┘   └───────▲───────────────┘
                               │ artifacts             │ ingest
                 ┌─────────────┴───────────────────────┴─────────────┐
                 │  Offline pipeline (training/, scripts/)           │
                 │  Colab fine-tune → eval harness → MLflow runs →   │
                 │  export ONNX + push embeddings to Qdrant          │
                 └───────────────────────────────────────────────────┘
```

**Key structural decisions:**

1. **The API is the product; Streamlit is just a client.** All model/retrieval logic moves out of `app/streamlit_app.py` into service-layer modules that FastAPI exposes; the UI calls the API over HTTP.
2. **Two models coexist, honestly.** The fine-tuned transformer (ONNX, CPU inference) is primary. TF-IDF+LR stays as a documented fast/explainable fallback (`model=baseline`). The measured comparison is a first-class feature of the repo.
3. **Qdrant replaces the `.npy` matrix.** One collection serves all three query types.
4. **Offline/online split.** Training and indexing are offline jobs producing versioned artifacts; the serving stack only loads artifacts. MLflow tracks runs; `models/registry.json` pins the deployed version.
5. **Repo grows, doesn't restart.** Existing `src/` modules survive for the baseline path; new dirs: `api/`, `training/`, `scripts/`, `tests/`, `docker/`. Notebooks remain as the research record.

## 3. Components

### 3.1 FastAPI service (`api/`)

All endpoints under `/v1`, Pydantic-validated:

| endpoint | request | response |
|---|---|---|
| `POST /v1/predict` | `{lyrics: str}` (1–10,000 chars); query params `explain` (default true), `model` (`transformer` default \| `baseline`) | `{mood, confidence, probabilities, explanation: [{token, weight}] \| null, model_version, warnings}` |
| `GET /v1/search` | `?q=<3–200 chars>&limit=10&mood=<optional>` | `{results: [{title, artist, mood, score}], query_embedding_ms, total_ms}` |
| `GET /v1/songs` | `?title=...&artist=...` (fuzzy) | single match: song + full mood analysis + 5 similar; multiple: ranked candidates |
| `GET /health` | — | `{status, model_loaded, qdrant_ok, model_version}` |
| `GET /metrics` | — | Prometheus text format: request counts, latency histograms, error counters per endpoint |

Cross-cutting:

- Consistent error envelope `{error: {code, message}}` with 400/404/422/429/503.
- `slowapi` rate limiting (30 req/min/IP) with `Retry-After` on 429.
- Request-ID middleware + structured JSON logging (`structlog`); raw lyrics are never logged.
- CORS configured for the Streamlit client.
- Artifacts and Qdrant client load once at startup via FastAPI `lifespan`; inference runs in a thread pool so the event loop stays free under concurrent load.

### 3.2 Model plan (`training/`)

- **Primary model**: fine-tune `distilbert-base-uncased` (66M params) with a 5-class head on the 76k labeled songs. Colab free GPU, ~1–2 hrs/run. Recipe: max_len 256, 2–3 epochs, class-weighted loss (Hype ≈ 55% of corpus), early stopping on val macro F1. Same 80/10/10 stratified split, `random_state=42`, as the baseline — numbers directly comparable to 0.432 / 0.371.
- **Export**: best checkpoint → ONNX with int8 dynamic quantization (~65MB; ~20–50ms/song on laptop CPU). Serving depends only on `onnxruntime` + tokenizer, never the training stack or a GPU.
- **Explanations**: transformer → token-level SHAP (text masker on the ONNX model), capped at top-10 tokens with truncated input to bound latency; skippable via `explain=false`. Baseline keeps exact `LinearExplainer` SHAP. Docs frame this as the accuracy↔explainability trade-off.
- **Tracking**: MLflow with local file backend. Every training/eval run logs params, metrics, confusion matrix artifact. `models/registry.json` pins the served artifact version + its data hash.
- **Eval harness** (`training/evaluate.py`): one command; runs any registered model against the frozen test split; emits accuracy, macro F1, per-class P/R, confusion matrix, and a markdown report. CI runs it on the baseline as a smoke test.

### 3.3 Qdrant schema

Collection `songs`:

- **Vector**: 384-d, `all-MiniLM-L6-v2` (unchanged), cosine distance.
- **Payload**: `{song_id, title, artist, mood, valence, energy, lyrics_excerpt (~300 chars)}`; payload indexes: `mood` keyword, `title`/`artist` full-text.
- **Ingest**: `scripts/index_corpus.py` — idempotent batch upserts from the processed dataset; reuses the existing embedding cache where valid.
- Full lyrics do **not** enter Qdrant (copyright + size); they are read from the local processed store (parquet) by `song_id` when needed.

Query-type mapping:

- similar-songs → vector search + `mood` filter (today's behavior, indexed)
- NL mood search → embed query text → vector search (+ optional mood filter)
- title/artist lookup → payload full-text match → fetch lyrics by id → run predict pipeline

## 4. Data Flow

### Request-time

- **`POST /v1/predict`**: validate → `clean_text()` (baseline) / tokenizer (transformer) → ONNX session in thread pool → softmax → optional SHAP attribution → response with `model_version`. Latency targets: p50 < ~150ms without explanation, < ~1.5s with.
- **`GET /v1/search`**: validate query → MiniLM encode in thread pool (~10ms) → Qdrant vector search (optional mood filter) → hydrate display fields from payload. Only titles/artists/excerpts/scores leave the server.
- **`GET /v1/songs`**: Qdrant full-text match on title/artist → multiple hits: ranked candidates; single hit: fetch full lyrics by `song_id` locally → predict pipeline + similar-songs → combined response.

### Offline

```
SpotGenTrack CSVs ──prep──► songs_labeled.parquet (versioned data hash)
songs_labeled ──Colab fine-tune──► checkpoint ──export──► model.onnx + tokenizer
                     │                                        │
                     └──► MLflow run (params/metrics) ──► registry.json pins version
songs_labeled ──index_corpus.py──► Qdrant collection (idempotent upserts)
```

Labeling logic (valence/energy thresholds, gap-zone drop) stays exactly as-is this month.

## 5. Error Handling

- **Input boundary**: Pydantic → 422 with field detail; empty/whitespace lyrics → 400; rate limit → 429 + `Retry-After`.
- **Degraded modes are explicit**: Qdrant down → `/predict` still works; `/search` and `/songs` return 503 `{"code": "retrieval_unavailable"}`; `/health` reports `qdrant_ok: false`. Missing/corrupt model artifact → fail fast at startup with a readable error naming the artifact path.
- **Explanations are non-fatal**: SHAP error/timeout → prediction returned with `explanation: null` + logged warning.
- **Non-English input**: prediction still returned; cheap heuristic adds `warnings: ["input may be non-English"]` (honest handling of the known `clean_text` Latin-script limitation).
- Every error path logs structured context (request ID, endpoint, input length — never raw lyrics) and increments a Prometheus error counter.

## 6. Testing

pytest, run in CI (GitHub Actions) on every push:

1. **Unit**: preprocessing, label derivation, ranking math, registry loading — pure-Python, fast.
2. **API integration**: FastAPI `TestClient` with a fake model + in-memory Qdrant substitute behind small interfaces — every endpoint's happy path, validation errors, and degraded modes. No heavy artifacts in CI.
3. **Contract smoke test** (local, pre-release): `docker compose up` + script hitting every real endpoint with real artifacts, asserting response shapes and latency budgets.
4. **Model quality gate**: eval harness asserts the deployed model beats the majority-class baseline on macro F1 (runs locally where artifacts exist).

## 7. Phasing (4 part-time weeks)

| week | deliverable | end state |
|---|---|---|
| 1 — spine | `api/` with `/predict` (baseline model), error contract, tests, Dockerfile + compose with Qdrant, `index_corpus.py` | dockerized API answering real predict queries |
| 2 — model | Colab fine-tune → eval harness + MLflow → ONNX export → served as primary | measurably better model in the production path; comparison table in README |
| 3 — real queries | `/search`, `/songs`, rate limiting, `/metrics`, Streamlit rewired as API client | all three query types live |
| 4 — polish & ship | CI pipeline, HF Spaces demo, README rewrite (architecture, before/after metrics, latency numbers) | shippable portfolio repo + live demo link |

**Stretch (only if week 4 has room):** LLM-assisted relabeling experiment on a 2–5k song subset using free-tier LLM access, reported as a label-quality ablation.

## 8. Definition of Done

- `docker compose up` → all three query types work end-to-end.
- CI green: lint + unit + API integration tests.
- README shows old-vs-new model metrics produced by the eval harness, plus measured latency.
- Live free-tier demo (HF Spaces) linked from the README.

## 9. Risks & Mitigations

| risk | mitigation |
|---|---|
| Fine-tune doesn't beat baseline by much (label noise ceiling) | Report honestly; the eval harness + analysis is itself the portfolio value. Baseline stays served either way. |
| Colab free-tier session limits interrupt training | Checkpoint every epoch to Drive; runs are 1–2 hrs, well within limits. |
| SHAP-on-transformer latency too high | Cap tokens/input length; `explain=false` path; baseline model offers exact-SHAP alternative. |
| HF Spaces free tier can't run api + Qdrant + UI together | Demo Space runs the API with an embedded/local Qdrant instance (single container mode); full compose stack remains the local/canonical deployment. |
| Scope creep past one month | Weeks are independently shippable; cut from the end (week 4 polish compresses; stretch goal drops first). |

## 10. Conventions Carried Forward

- AI-attribution docstring blocks continue in all new modules (per `ATTRIBUTION.md`).
- `random_state=42` everywhere splits/sampling occur.
- `docs/rubric-mapping.md` is legacy (course artifact) — frozen, not extended.
- README metrics must be regenerated by the eval harness, never hand-edited.
