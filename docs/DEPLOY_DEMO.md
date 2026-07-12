# Deploying the demo to Streamlit Community Cloud

This runbook takes the LyricMood demo from a local checkout to a public URL on
**Streamlit Community Cloud** (share.streamlit.io) — free, no card, deploys a
branch of the public GitHub repo. It's a one-time, ~20-minute procedure.

> **Why not Hugging Face Spaces?** The original plan was a single-container HF
> Space (`docker/Dockerfile.spaces`). In July 2026 HF Spaces' free tier dropped
> the Docker/Gradio SDKs, so that route is gone for the free tier. The
> `Dockerfile.spaces` image is kept in the repo as the **paid-tier / self-host**
> option (it still runs uvicorn + Streamlit + embedded Qdrant in one container);
> this runbook is the free-hosting path.

## How the SCC demo works

Streamlit Community Cloud only runs `streamlit run <entrypoint>` against the
repo-root `requirements.txt` — no container, no compose stack, no separate
Qdrant service. So the entrypoint **`demo_entry.py`** reconstructs the whole
runtime in one process:

1. downloads the demo bundle (model artifacts + a file-based Qdrant) from a
   public HF **model** repo at first boot, cached to disk by `huggingface_hub`;
2. rewrites the bundled `registry.json` so the transformer `dir` becomes an
   absolute path inside the downloaded bundle (`registry_runtime.json`);
3. starts the FastAPI app **in-process** on `127.0.0.1:8000` (daemon thread) and
   waits for `/health`;
4. hands off to the unmodified `app/streamlit_app.py`, which talks to that
   in-process API over HTTP exactly as it does locally.

### What works on the demo (and what doesn't)

The bundle deliberately **excludes the full lyrics**
(`data/processed/songs_labeled.csv`) for copyright (redistributing full lyrics
is not ours to do) and size reasons — the Qdrant payloads carry only ≤300-char
excerpts.

Consequence, **by design**:

| capability | on the demo | why |
|---|---|---|
| `POST /v1/predict` (paste lyrics → mood + SHAP) | works | model artifacts are bundled |
| `GET /v1/search` (free-text mood search) | works | excerpts + embeddings are bundled |
| `POST /v1/similar` (paste lyrics → 5 songs) | works | same |
| `GET /v1/songs` — single match → **full analysis** | **503 `lyrics_unavailable`** | needs the absent full-lyrics store |

The UI never calls `/v1/songs`, so demo users never hit the 503 — it's the API's
existing degraded-mode contract, not a bug.

## Step 0 — Prerequisites

- A [Hugging Face account](https://huggingface.co/join) (free) to host the model
  bundle, and a GitHub account (the repo is already public).
- The Hugging Face CLI, logged in with a **write** token. No `git-lfs` needed —
  `hf upload` streams large files itself.

```bash
pip install -U huggingface_hub      # provides the `hf` CLI (>=1.23)
hf auth login                       # paste a WRITE token (hf.co/settings/tokens)
```

- A local checkout with all pipeline artifacts already built (see
  [SETUP.md](../SETUP.md)) — the bundle is assembled from them.

## Step 1 — Build the demo bundle locally

From the project root, with your venv active:

```bash
python scripts/build_demo_bundle.py          # writes ./demo/ (gitignored)
```

This copies the model artifacts and indexes the full corpus into a file-based
Qdrant with excerpt-only payloads. Expect a `demo/` directory of **~493 MB**
(`models/` + `qdrant_local/`). The script exits non-zero and lists what's
missing if any source artifact is absent. (Already done for this checkout.)

## Step 2 — Upload the bundle to a public HF model repo

Create a **public model** repo and upload the whole `demo/` directory into it.
(The `hf` CLI renamed its subcommands in the 1.x line — these are the verified
working forms as of `hf` 1.23; `hf repo create` still works but warns it is
deprecated in favor of `hf repos create`.)

```bash
# create the repo (public model repo)
hf repos create evelindsayyy/lyricmood-demo --repo-type model --public

# upload the bundle: REPO_ID  LOCAL_PATH  PATH_IN_REPO
hf upload evelindsayyy/lyricmood-demo demo/ . --repo-type model
```

`demo_entry.py` reads the repo id from the `LYRICMOOD_DEMO_BUNDLE_REPO` env var
and defaults to `evelindsayyy/lyricmood-demo`, so if you use that name there is
nothing else to configure. (To use a different repo, set that env var in the SCC
app's **Advanced settings → Secrets**.)

## Step 3 — Create/refresh the `demo` branch and push it

SCC only reads the repo-root `requirements.txt`, but on `main` that file is the
notebook/research requirements. The `demo` branch's **only** divergence from
`main` is that root `requirements.txt` is replaced by `requirements-demo.txt`
(the union of the API + UI deps plus `huggingface_hub`). Regenerate it whenever
`main`'s requirements change:

```bash
git checkout main
git branch -D demo 2>/dev/null || true       # drop any stale demo branch
git checkout -b demo
cp requirements-demo.txt requirements.txt
git commit -am "demo: scc root requirements"
git push -u origin demo --force-with-lease
git checkout main
```

## Step 4 — Deploy on Streamlit Community Cloud

On https://share.streamlit.io → **New app** → **Deploy from GitHub**:

- **Repository:** `evelindsayyy/lyrics_mood_predictor`
- **Branch:** `demo`
- **Main file path:** `demo_entry.py`

Click **Deploy**. First boot downloads the 493 MB bundle (~2–4 min) then loads
the models (~10–20 s) before the UI appears.

**Notes**

- **Sleep:** a Community Cloud app sleeps after ~7 days of inactivity and wakes
  on the next visit (first request after a sleep re-downloads nothing — the
  bundle is cached in the container image layer until the container is recycled).
- **Resources:** the free tier gives ~2.7 GB RAM; the demo (ONNX models +
  file-based Qdrant + Streamlit) fits comfortably.
- **Rate limit:** every visitor reaches the in-process API from localhost, i.e.
  one shared bucket, so `demo_entry.py` sets `LYRICMOOD_RATE_LIMIT=240/minute`
  (the default 30/min would throttle the whole demo at ~15 analyses/min).

## Step 5 — Wire the URL into the README

Paste the SCC app URL into the live-demo line at the top of
[README.md](../README.md):

```markdown
**[Live demo →](https://<your-app>.streamlit.app)**
```

Commit that change to the project repo. Done.
