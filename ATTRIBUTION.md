# AI Tool Attribution

I used **Claude (Anthropic)** as a coding collaborator on this project, including agentic-mode code generation, which is permitted under the course's AI policy. The policy frames the best use of AI as collaborative — *"using AI to accelerate implementation, syntax lookup, and debugging, while you retain the critical role of system and model designer, planning experiments, and critically evaluating results."* That's how I worked: I designed the system, planned the experiments, evaluated the results, and used Claude to accelerate the implementation.

Below is an honest breakdown of where AI help was material vs. where I wrote things myself.

## Notebooks

- **`notebooks/01_eda.ipynb`** — **I wrote this myself**, with light AI suggestions (asking "what's the cleanest way to plot the valence-energy scatter with overlay lines" kind of thing). EDA, lyric-length distribution, valence-vs-energy scatter, mood-region application, class-distribution check, gap-zone count, the labeled-data save, and the two preprocessing experiments (class-weight and gap-zone A/Bs) with their before/after macro F1 numbers. The narrative interpretation is mine.
- **`notebooks/02_modeling.ipynb`** — **Roughly half and half**. The 80/10/10 split scaffolding and the two baseline functions are mine. The 7-config hyperparameter sweep loop and the LR-vs-NB head-to-head comparison were written by Claude based on my design (I picked the C and α grids, the metric set, and which model to save). I read the sweep results and picked the winning config myself.
- **`notebooks/03_evaluation.ipynb`** — **Written by Claude based on my direction.** I designed each experiment — the confusion matrix analysis, the SHAP-on-misclassified-examples loop, the four-category edge-case probe, the two improvement iterations, the SHAP faithfulness deletion test, and the retrieval semantic-quality precision@5 self-check. I picked the metrics, sample sizes (100 / 200), and the random `Σ P(c)²` baseline. Claude wrote the loops and the formatted output blocks. **The narrative analysis is mine** — the *why the model fails* causal discussion (vocabulary overlap + majority-class gravity + label noise from valence/energy thresholding) and the diagnostic recipe ("look at SHAP top words to tell label noise from a real model error") came from me reading the misclassified examples and synthesizing what I saw.

## Source modules in `src/`

**Written by Claude based on my specifications.** I gave Claude the function signatures, the parameter values, and the architectural decisions; Claude wrote the function bodies. I read, tested, and validated each module:

- `src/preprocess.py` — I specified the 5-mood taxonomy, the valence/energy threshold values (0.3/0.6 and 0.4/0.6), the gap-zone filtering rule, and the `clean_text` cleaning behavior (strip Genius section markers, lowercase, remove punctuation, drop sklearn stopwords). Claude wrote the function bodies.
- `src/features.py` — I specified the TF-IDF settings (defaulting to `max_features=20000`, `ngram_range=(1,2)`, `min_df=3`, `sublinear_tf=True` after my second-pass tuning) and the joblib save behavior. Claude wrote the wrappers.
- `src/classify.py` — I specified the 80/10/10 stratified split with `random_state=42`, the four-metric evaluation bundle (accuracy, macro F1, per-class P/R), and the function signatures. Claude wrote the bodies.
- `src/recommend.py` — I specified the MiniLM model (`all-MiniLM-L6-v2`), the disk-cache strategy, the L2-normalize-at-embed-time convention (Claude proposed this optimization in a concept-explanation conversation), the mood-filter behavior, and the cosine-similarity ranking. Claude wrote the bodies.
- `src/explain.py` — I specified using SHAP `LinearExplainer` (because the model is linear and this gives exact Shapley values), the background-sample-from-training strategy, and the input-vocabulary filter so "top negatives" only show words actually in the input. Claude wrote the bodies and the version-defensive `isinstance` branch (which came out of a debugging session after I hit a SHAP shape mismatch).

## API service — `api/` and `scripts/index_corpus.py`

**Written by Claude based on my design spec** ([docs/superpowers/specs/2026-07-09-industrial-elevation-design.md](docs/superpowers/specs/2026-07-09-industrial-elevation-design.md)). I designed the FastAPI app-factory structure, the DI/lifespan approach, the error contract, and the Qdrant indexing scheme; Claude implemented them. I reviewed and tested every file, including running the full Docker Compose stack end-to-end (76,595 songs indexed) and the 33-test pytest suite:

- **`api/`** (`config.py`, `schemas.py`, `errors.py`, `logging_setup.py`, `deps.py`, `main.py`, `routes/`, `services/`) — the FastAPI service: settings, request/response schemas, the `{"error": {code, message}}` contract, logging setup that never logs raw lyrics, dependency injection, the `create_app()` factory, health/predict routes, and the model/retrieval service wrappers.
- **`scripts/index_corpus.py`** — the one-time/idempotent script that populates the Qdrant collection from the processed corpus.

### Week-3: real query endpoints, rate limiting, metrics, and the streamlit rewrite

**Written by Claude based on my design spec** ([docs/superpowers/plans/2026-07-10-week3-real-queries.md](docs/superpowers/plans/2026-07-10-week3-real-queries.md)). I specified the contracts and the constraints; Claude implemented them and worked through the FastAPI/slowapi/qdrant version-compatibility issues that came up along the way. I reviewed and tested every file, including running the full suite (100 tests) and smoke-testing `/v1/search`, `/v1/similar`, `/v1/songs`, `/metrics`, and the three-container Docker Compose stack (ui + api + qdrant) end-to-end:

- **`scripts/export_minilm_onnx.py`** — the one-time export of `all-MiniLM-L6-v2` to ONNX, with a parity check against the original sentence-transformers model before the export is trusted for serving.
- **`api/services/embedder.py`** — the torch-free ONNX query embedder (`Embedder` protocol), reimplementing masked mean-pooling + L2 normalization in numpy so query vectors land in the same space as the corpus embeddings.
- **`api/services/retrieval.py`** (search + `find_song`) — vector search and fuzzy title/artist lookup against Qdrant.
- **`api/services/songs.py`** (`LyricsStore`) — the lyrics lookup keyed by the indexer's `song_id`, backing `GET /v1/songs`.
- **`api/ratelimit.py`** — I specified 30 req/min/IP with 429 + `Retry-After`; Claude implemented it, and when slowapi's own middleware turned out unable to resolve routes registered through FastAPI's included-router wrapper (silently enforcing nothing), Claude diagnosed it and wrote `RateLimitMiddleware` as a working replacement over the same `slowapi.Limiter`.
- **`api/metrics.py`** — the Prometheus `/metrics` endpoint and request instrumentation, with the same included-router quirk worked around so route-template labels stay correct instead of collapsing to the un-prefixed path.
- **`app/streamlit_app.py`** — rewritten from a monolithic model-loading app into a pure API client (zero ML imports) that talks to the API over HTTP via `LYRICMOOD_API_URL`, with a `ui` Docker service added alongside `api` and `qdrant`. I directed the pivot to the client/server split; Claude wrote the rewrite and the `docker/Dockerfile.ui` container.

I also caught and had Claude fix an honesty issue from a prior review: `requirements-api.txt` pinned `qdrant-client>=1.9`, but `api/services/retrieval.py` calls `query_points`, which needs client `>=1.10` — the floor is now `>=1.10`.

## Transformer training + multi-model serving — `training/` and `api/services/{registry,transformer}.py`

**Written by Claude based on my design spec** ([docs/superpowers/specs/2026-07-09-industrial-elevation-design.md](docs/superpowers/specs/2026-07-09-industrial-elevation-design.md), §3.2). I specified the fine-tuning recipe (`distilbert-base-uncased`, max_len 256, 2-3 epochs, class-weighted loss, early stopping on val macro F1), the identical-split discipline with the notebooks (`random_state=42`), the int8 dynamic ONNX quantization export, the `--smoke` mode for a fast CPU-only pipeline check, the eval harness contract (frozen test split, quality gate vs. majority-class baseline, markdown report), and the registry-driven multi-model serving design (`models/registry.json` pins loaded models + default, `?model=` query param selects per-request). Claude implemented them; I ran the `--smoke` check and the `--model baseline` eval end-to-end to verify both work before this runbook was written:

- **`training/finetune_distilbert.py`** — the fine-tuning + ONNX export script (Colab-targeted, CPU-testable via `--smoke`).
- **`training/evaluate.py`** — the eval harness: frozen-split scoring, quality gate, markdown report, optional MLflow logging.
- **`api/services/registry.py`** — loads and validates `models/registry.json`.
- **`api/services/transformer.py`** — the ONNX/onnxruntime serving wrapper implementing the same `MoodModel` protocol as the baseline, so `?model=transformer` is a drop-in swap.

The actual Colab fine-tune run, the transformer-vs-baseline comparison, and the registry promotion decision are mine to make (see [training/README.md](training/README.md)) — not yet done as of this commit.

## Week-4: CI, single-container demo, deploy runbook

**Written by Claude based on my design spec** ([docs/superpowers/plans/2026-07-11-week4-ship-it.md](docs/superpowers/plans/2026-07-11-week4-ship-it.md), tracing the Week-4 row of [the elevation spec](docs/superpowers/specs/2026-07-09-industrial-elevation-design.md)). I specified the deliverables and the hard constraints — free-tier only, tests stay artifact-free and network-free, `requirements-api.txt` gains nothing, full lyrics never enter any public artifact (excerpts ≤300 chars only), and the demo's degraded-mode `/v1/songs` 503 must be a documented design choice, not a regression. Claude implemented them; I reviewed and ran the full suite (106 tests), the ruff lint, and the single-container demo image locally end-to-end before this runbook was written:

- **`.github/workflows/ci.yml`** — lint (`ruff check .`, pinned to the locally-verified version), the pytest suite, and api/ui Docker builds, on every push to `main` and every PR. I specified the three-job shape and that CI must run on the artifact-free suite.
- **`ruff.toml`** — the lint config; frozen notebooks are excluded so generated cell IDs don't trip the linter.
- **`scripts/build_demo_bundle.py`** — assembles the `demo/` bundle (model artifacts + a file-based local-path Qdrant with excerpt-only payloads). I specified the excerpt-only constraint, the registry-dir rewrite (so the transformer path resolves inside the container), and that it must reuse the cached `corpus_embeddings.npy` and never re-embed. It reuses the indexer's functions directly.
- **`docker/Dockerfile.spaces`** + **`docker/spaces_launcher.sh`** — the single-container image (uvicorn :8000 + Streamlit :7860 in one process supervisor) with every `LYRICMOOD_*` artifact path pointed at the in-bundle copies; `labeled_songs_path` is left absent on purpose so the lyrics store degrades.
- **`api/config.py` / `api/services/retrieval.py` / `api/main.py`** — the additive `qdrant_path` setting and `QdrantRetrieval.local()` that switch retrieval to serverless local-path Qdrant when the path is set. I specified that this be additive — no existing contract, route, or settings name changes.
- **Deferred-minors cleanup** — the small week 1–3 fixes I had flagged for later (empty-embed guard, registry `kind` validation, whitespace-only search-query guard, `--limit` validation, E741 renames) with their tests; no public contract changed.
- **`docs/DEPLOY_SPACES.md`** and the **`README.md` rewrite** (architecture diagram, API table, the "class project → production system" timeline) — drafted by Claude from my notes and the verified repo state; I edited, validated, and own the content. The LLM-assisted relabeling idea is documented as future work.

The Colab fine-tune, the Space deployment itself, and filling the README's live-demo URL are mine to do (post-deploy) — the runbook exists so I can.

## Streamlit app — `app/streamlit_app.py`

**Written by Claude based on my visual design.** I created the design upfront — `docs/design/LyricMood Minimal.html` (the visual mock), `app/static/lyricmood.css` (the design tokens and component styles), and `docs/design/DESIGN_HANDOFF.md` (the spec doc) — and gave that to Claude as input. Claude wrote the Python implementation and the CSS overrides that re-skin Streamlit's built-in widgets to match (chipbar layout, raw-HTML SHAP horizontal bar chart, song-list grid, `set_mood_accent()` helper for swapping the active mood color). I integrated, tested, iterated, and own the `@st.cache_resource` data-loading strategy and the prediction → SHAP → retrieval data flow.

## Documentation

`README.md`, `SETUP.md`, and the files in `docs/` were drafted by Claude from my notes, decisions, and observed results. I edited, validated, and own the content.

## Specific debugging help from Claude

- *"All my cosine similarities for one query are exactly 1.0"* — Claude suggested checking the corpus for duplicate raw lyrics. I verified with pandas, traced it to a Kaggle scraper artifact, documented in `docs/findings.md`.
- *"Korean lyrics return Hype with low confidence"* — Claude pointed to my `clean_text()` regex stripping non-Latin characters, which leaves an empty string and reduces the model to the class prior. Documented as a Task 10 edge-case finding.
- A SHAP shape mismatch — Claude explained the multiclass output-shape variation across shap versions, which led to the `isinstance` branch in `src/explain.py`.
- A CSS-loading bug — Claude spotted from a broken-page screenshot that my design file had a literal `</style>` tag inside its comment header, which was closing my wrapping `<style>` tag early.

## Concept explanations Claude provided

- Why `LinearExplainer` is exact for linear models (closed-form Shapley values).
- Why `normalize_embeddings=True` lets cosine similarity become a plain dot product.
- How `class_weight='balanced'` interacts with `max_iter` in sklearn LR.

## What I'm responsible for — design + decisions + evaluation

These are my contributions to the project, none of which were AI-generated:

- **System design**: the 5-mood taxonomy grounded in Russell's circumplex, the valence/energy threshold values, the gap-zone filter, the architectural choice to keep TF-IDF (for classification + SHAP) and MiniLM (for retrieval) in disjoint `src/` files. The decision to use logistic regression (so SHAP is exact) instead of a neural network.
- **All hyperparameter values**: `C=1.0`, L2 penalty, `class_weight='balanced'`, `ngram_range=(1,2)`, `max_features=20000`, `min_df=3`, `sublinear_tf=True`. These came out of sweeps that I ran and read.
- **Experimental design**: which preprocessing experiments to run (class-weight, gap-zone), which improvement iterations to test (ngrams, class-weighting), the metrics and sample sizes for the SHAP faithfulness and retrieval self-check experiments.
- **Result interpretation**: reading every misclassified test example and writing the *why the model fails* causal discussion, the diagnostic recipe linking SHAP top words to label noise vs. model error, the per-class word-count audit, and the data-quality findings (duplicate lyrics, standup comedy in the Sad class).
- **Frontend visual design**: the mock, the tokens, the typographic and palette choices.
- **Project plan**: the rubric mapping in `docs/rubric-mapping.md`, the per-task scope, and the order of work.
- **Validation**: re-executing all three notebooks end-to-end multiple times, querying the deployed Streamlit app with diverse inputs, verifying every result.

I'm fully responsible for everything in this repository — "but the AI said so" wouldn't be a valid defense. I read every line of code in this project and can defend it in the technical walkthrough.

## Other tools

- **VSCode + Jupyter extension** for editing.
- **sklearn / shap / sentence-transformers / Streamlit** official docs for API specifics.
- **No other code-generation AI** (no Copilot, Cursor, etc.).
