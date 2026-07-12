# Deploying the demo to Hugging Face Spaces

This runbook takes the single-container demo (`docker/Dockerfile.spaces`) from a
local checkout to a public URL on Hugging Face Spaces' **free CPU basic** tier.
It's a one-time, ~20-minute procedure; a Hugging Face account is the only
prerequisite you don't already have.

The demo image runs both processes in one container: `uvicorn` on `:8000`
(internal) and Streamlit on `:7860` (the port the Space exposes), wired together
by `docker/spaces_launcher.sh`. All model artifacts and a file-based serverless
Qdrant travel inside the image as the `demo/` bundle — there is no separate
vector-DB service to run.

## What works on the demo (and what doesn't)

The bundle deliberately **excludes the full lyrics** (`data/processed/songs_labeled.csv`)
for two reasons: copyright (redistributing full lyrics is not ours to do) and
size (~150 MB). The Qdrant payloads carry only ≤300-char excerpts.

Consequence, **by design**:

| capability | on the demo | why |
|---|---|---|
| `POST /v1/predict` (paste lyrics → mood + SHAP) | works | model artifacts are bundled |
| `GET /v1/search` (free-text mood search) | works | excerpts + embeddings are bundled |
| `POST /v1/similar` (paste lyrics → 5 songs) | works | same |
| `GET /v1/songs` — ambiguous title → **candidate list** | works | no lyrics needed to list matches |
| `GET /v1/songs` — single match → **full analysis** | **503 `lyrics_unavailable`** | needs the absent full-lyrics store |

The 503 is the API's existing degraded-mode contract, not a bug — the lyrics
store is simply never loaded in the container.

**Free-tier caveat:** a free Space **sleeps after ~48 h of inactivity** and cold-starts
on the next visit (the first request after a sleep takes ~30–60 s while the
container boots). This is expected and fine for a portfolio demo.

## Prerequisites

- A [Hugging Face account](https://huggingface.co/join) (free).
- `git` and `git-lfs` installed locally (`git lfs install` once per machine).
- A local checkout with all pipeline artifacts already built — the demo bundle
  is assembled from them, and `build_demo_bundle.py` refuses to run if any are
  missing. You need:
  - `models/best_classifier.pkl`, `models/tfidf_vectorizer.pkl`
  - `models/transformer/` (ONNX int8 export) and `models/embedder/` (ONNX MiniLM)
  - `models/registry.json`
  - `models/corpus_embeddings.npy` (cached embeddings — the builder never re-embeds)
  - `data/processed/songs_labeled.csv` (used **only** to build the local Qdrant; not copied into the bundle)
  - `SpotGenTrack/Data Sources/spotify_artists.csv` (for artist-name resolution)

  See [SETUP.md](../SETUP.md) if any of these don't exist yet.

## Step 1 — Prerequisites + log in to Hugging Face

```bash
# git-lfs (large-file support — the bundle is ~500MB)
brew install git-lfs           # macOS; on Linux: apt install git-lfs
git lfs install                # one-time git hook setup

# hugging face CLI (the old `huggingface-cli` command is deprecated)
pip install -U huggingface_hub
hf auth login                  # paste a token with WRITE access (hf.co/settings/tokens)
```

## Step 2 — Create a Docker Space

On https://huggingface.co/new-space:

- **Owner / Space name:** e.g. `your-username/lyricmood`
- **SDK:** **Docker** → **Blank** template
- **Hardware:** **CPU basic** (free)
- **Visibility:** Public

Create it. The Space starts as an empty git repo you'll push into.

## Step 3 — Build the demo bundle locally

From the project root, with your venv active:

```bash
python scripts/build_demo_bundle.py          # writes ./demo/ (gitignored in this repo)
```

This copies the model artifacts (rewriting `registry.json` so the transformer
dir points inside the bundle) and indexes the full corpus into a file-based
Qdrant with excerpt-only payloads. Expect a `demo/` directory of **~493 MB**
(`models/` + `qdrant_local/`). The script exits non-zero and lists what's
missing if any source artifact is absent.

## Step 4 — Assemble the Space repo

Clone the Space and copy in exactly what the image needs. `Dockerfile.spaces`
`COPY`s `src/ api/ app/ demo/`, both requirements files, and
`docker/spaces_launcher.sh` (its `CMD`), so preserve that layout:

```bash
git clone https://huggingface.co/spaces/your-username/lyricmood
cd lyricmood

# from your project checkout ($SRC = path to LyricsMoodPredictor):
cp "$SRC/docker/Dockerfile.spaces" Dockerfile          # HF looks for ./Dockerfile
mkdir -p docker
cp "$SRC/docker/spaces_launcher.sh" docker/
cp -r "$SRC/src" "$SRC/api" "$SRC/app" "$SRC/demo" .
cp "$SRC/requirements-api.txt" "$SRC/requirements-ui.txt" .
```

You do **not** copy `data/processed/`, the notebooks, `training/`, or the raw
dataset — none are part of the running demo. All artifact paths are injected as
env vars inside `Dockerfile.spaces` (`LYRICMOOD_MODEL_DIR=demo/models`,
`LYRICMOOD_QDRANT_PATH=demo/qdrant_local`, etc.), so there is nothing to
configure by hand.

Add a Space header to the top of `README.md` in the Space repo (HF requires the
front-matter to pick the port):

```yaml
---
title: LyricMood
sdk: docker
app_port: 7860
---
```

## Step 5 — Track large files with LFS, commit, push

The bundle holds binary model files and a Qdrant store — these must go through
git-lfs or the push will be rejected:

```bash
git lfs track "demo/**" "*.onnx" "*.pkl" "*.npy"
git add .gitattributes Dockerfile docker/ src/ api/ app/ demo/ \
        requirements-api.txt requirements-ui.txt README.md
git commit -m "deploy lyricmood single-container demo"
git push
```

## Step 6 — Wait for the build

The Space builds the Docker image and boots the container — **~10 min** for the
first build (installing `requirements-api.txt` + `requirements-ui.txt`). Watch
the **Logs** tab; when it shows Streamlit serving on `:7860`, the demo is live
at `https://your-username-lyricmood.hf.space`.

Quick smoke test once it's up:

```bash
curl -X POST https://your-username-lyricmood.hf.space/v1/predict \
  -H 'content-type: application/json' -d '{"lyrics": "we dance all night under the lights"}'
```

(The Streamlit UI proxies to the same in-container API, so if `/v1/predict`
answers, the UI works too.)

## Step 7 — Wire the URL into the README

Paste the Space URL into the live-demo line at the top of the project
[README.md](../README.md):

```markdown
**[Live demo →](https://your-username-lyricmood.hf.space)**
```

Commit that change to the project repo. Done.
