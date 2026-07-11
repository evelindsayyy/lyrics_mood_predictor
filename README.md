# LyricMood

A small web app that reads song lyrics, predicts the mood, explains *why*, and surfaces 5 other songs with a similar emotional profile.

## What it Does

LyricMood is a two-model system glued together by a Streamlit UI. You paste song lyrics into the web app and get three things back:

1. **A mood prediction** (one of *Hype, Romantic, Calm, Sad, Angry*) with a confidence score, produced by a logistic regression trained on TF-IDF features of ~76,000 labeled songs.
2. **A word-level explanation** of that prediction (interpretability via SHAP), pulling the top-10 words that pushed the model toward (or away from) the predicted mood — so the model isn't a black box.
3. **Five similar songs**, found by embedding the lyrics with a frozen MiniLM sentence-transformer and ranking the corpus by cosine similarity (filtered to the predicted mood).

Labels come from Spotify's audio features (valence + energy, cut into mood regions based on [Russell's circumplex model](#research-connections)), so the model is learning which lyrical patterns tend to go with which audio-derived moods.

## Quick Start

```bash
# 1. env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. get the dataset and the processed corpus (see SETUP.md for full details)
#    - download SpotGenTrack → SpotGenTrack/Data Sources/spotify_tracks.csv
#    - run notebooks/01_eda.ipynb to produce data/processed/songs_labeled.csv
#    - run notebooks/02_modeling.ipynb to produce models/best_classifier.pkl + tfidf_vectorizer.pkl
#    - run a one-liner to produce models/corpus_embeddings.npy

# 3. run the app — streamlit is a pure API client, so the api must be up too
#    (see step 4; or run `uvicorn api.main:app --reload` and then this)
streamlit run app/streamlit_app.py

# 4. run the full stack (api + vector db + web ui)
docker compose up --build          # ui :8501, api :8000, qdrant :6333
python scripts/index_corpus.py     # one-time corpus indexing
python scripts/export_minilm_onnx.py  # one-time query-embedder export

# then: open http://localhost:8501, or hit the API directly —
curl -X POST localhost:8000/v1/predict -H 'content-type: application/json' -d '{"lyrics": "..."}'
curl "localhost:8000/v1/search?q=rainy%20late%20night%20drive"
curl -X POST localhost:8000/v1/similar -H 'content-type: application/json' -d '{"lyrics": "...", "limit": 5}'
curl "localhost:8000/v1/songs?title=midnight"
```

Full step-by-step setup is in [SETUP.md](SETUP.md).

## Video Links

Both videos are stored in this repository under [`videos/`](videos/) and also linked below for direct viewing.

- **Project Demo** (3–5 min, non-technical): [`videos/demo.mp4`](videos/demo.mp4) · [YouTube](https://youtu.be/YAT78qsMW6o)
- **Technical Walkthrough** (5–10 min): [`videos/walkthrough.mp4`](videos/walkthrough.mp4) · [YouTube](https://youtu.be/saAWmiuv8NY)

> Before submission: record both videos, save them as `videos/demo.mp4` and `videos/walkthrough.mp4` in the repo, push, then replace each `PASTE_..._URL_HERE` placeholder with the public mirror link (YouTube unlisted / Loom / Google Drive / Vimeo).

## Evaluation

### Baseline vs. fine-tuned transformer

Two models serve behind the same API (`POST /v1/predict?model=baseline|transformer`), scored on the identical frozen test split (n=7,660) by `training/evaluate.py` — full reports in [`results/eval_baseline.md`](results/eval_baseline.md) and [`results/eval_transformer.md`](results/eval_transformer.md):

| model | test accuracy | test macro F1 | serving | explanations |
|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.432 | 0.371 | sklearn, ~ms | exact SHAP (`LinearExplainer`) |
| **DistilBERT fine-tune, ONNX int8 (default)** | **0.521** | **0.395** | onnxruntime CPU, torch-free | approximate SHAP (Text masker, capped) |

The transformer (2 epochs on a free Colab T4, class-weighted loss, early-stopped on val macro F1 = 0.408) wins on both headline metrics — most of the gain is Hype recall (0.46 → 0.68) and Sad, at the cost of some Calm recall (0.39 → 0.22). Both models remain served: the baseline is kept as the fast, exactly-explainable option, and the accuracy↔explainability trade-off is deliberate. The remaining ceiling is label noise from valence/energy thresholding, not model capacity — see the error analysis below.

Baseline recipe: logistic regression (C=1.0, L2, `class_weight='balanced'`) on TF-IDF features (unigrams + bigrams, 20k vocab cap, min_df=3, sublinear_tf). The first-pass sweep of 7 configs (4 LR × 3 MultinomialNB) is in [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb); a second-pass tuning of the TF-IDF knobs is in [notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb). Baseline detail:

| metric | value | notes |
|---|---|---|
| test accuracy | 0.432 | vs. 0.546 majority-class baseline |
| test macro F1 | 0.371 | vs. 0.141 majority-class, 0.201 random-weighted |
| per-class precision | Hype 0.73, Sad 0.33, Calm 0.27, Romantic 0.26, Angry 0.23 | |
| per-class recall | Angry 0.45, Hype 0.46, Romantic 0.40, Calm 0.39, Sad 0.36 | |

Macro F1 is the right metric here because Hype is ~55% of the corpus — overall accuracy is easy to game by just predicting Hype. The model beats the majority-class baseline's macro F1 by a factor of 2.5×.

Error analysis in [notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb) shows most residual errors cluster in three mood pairs (Romantic↔Hype, Angry↔Hype, Sad↔Calm) and are driven by shared genre vocabulary plus label noise from the valence-energy thresholding — not by the classifier itself being broken.

### Each project objective has a quantitative metric

| objective | metric | result |
|---|---|---|
| Predict mood from lyrics | test accuracy / macro F1 | 0.432 / 0.371 (vs. 0.546 / 0.141 majority-class) |
| SHAP explanations are *faithful* (not decorative) | mean confidence drop when top-5 SHAP words deleted; class-flip rate | 0.098 mean drop; 62/100 class flips on 100 correctly-classified test songs |
| MiniLM retrieval carries mood signal independently of the explicit mood filter | unfiltered mood-match precision@5 vs. random baseline | 0.478 vs. 0.354 random — **1.35× lift** on 200 corpus queries |

Full derivations in the *evaluation directly tied to project objectives* section of [notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb).

### Sample outputs

**Confusion matrix (test set, row-normalized):**

![confusion matrix](results/confusion_matrix.png)

The diagonal shows per-class recall. Brightest off-diagonal cells (Romantic→Hype, Angry→Hype, Calm↔Sad) are mood pairs that share lyrical vocabulary — analyzed in detail in `notebooks/03_evaluation.ipynb`.

**SHAP explanation for an Angry prediction:**

![SHAP angry](results/shap_angry.png)

Green bars = words pushing toward the predicted mood; red = pushing away. The model exposes its own reasoning, so a user can sanity-check whether a prediction is being driven by sensible vocabulary or noise.

## Research Connections

This project leans on three pieces of prior work:

- **Russell, J. A. (1980). *A circumplex model of affect.*** *Journal of Personality and Social Psychology, 39(6), 1161–1178.* — Motivates the 2-D valence/energy mood space. The 5 mood labels (Hype, Romantic, Calm, Sad, Angry) are named regions in Russell's circumplex, cut by thresholding Spotify's `valence` and `energy` scalars.
- **Hu, X., & Downie, J. S. (2010). *When lyrics outperform audio for music mood classification: A feature analysis.*** *Proceedings of ISMIR 2010.* — Shows lyric-derived features can beat audio features on mood classification. Motivates using lyrics (not audio features) as the model input, while letting audio features serve only as label proxies.
- **Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.*** *Proceedings of EMNLP 2019.* — Sentence-transformer architecture used for the retrieval half of the app. Specifically I use the pretrained `all-MiniLM-L6-v2` model, which outputs 384-d vectors cheap enough to index the full ~80k-song corpus.

## Individual Contributions

This is a **solo project** — all design, implementation, analysis, and writing was done by me. There were no other contributors.

Specifically, I was responsible for:

- **Project planning & ML design** — the rubric mapping (see [docs/rubric-mapping.md](docs/rubric-mapping.md)), the 5-mood taxonomy, valence/energy threshold values, the gap-zone filter, the choice of TF-IDF + Logistic Regression for classification (so SHAP `LinearExplainer` can run exactly), and MiniLM for retrieval (semantic similarity).
- **Implementation** — every `src/` module (`preprocess.py`, `features.py`, `classify.py`, `recommend.py`, `explain.py`), all three Jupyter notebooks, and the Streamlit app.
- **Hyperparameter selection** — `C=1.0`, `class_weight='balanced'`, `ngram_range=(1,2)`, `max_features=20000`, `min_df=3`, `sublinear_tf=True`.
- **Analysis & writing** — the error-analysis discussion in `03_evaluation.ipynb`, the edge-case interpretation, the improvement-iteration writeup, and all the project documentation (this README, `SETUP.md`, `ATTRIBUTION.md`).
- **Frontend visual design** — `docs/design/LyricMood Minimal.html`, `app/static/lyricmood.css`, the design tokens (palette, fonts, spacing).

AI-tool usage (Claude) is documented separately and in detail in [ATTRIBUTION.md](ATTRIBUTION.md).

## More Documentation

Additional documentation lives in [`docs/`](docs/):

- [`docs/architecture.md`](docs/architecture.md) — system architecture diagram and the rationale for the two-pipeline split (TF-IDF for classification, MiniLM for retrieval).
- [`docs/rubric-mapping.md`](docs/rubric-mapping.md) — maps each ML rubric checkbox to the file or notebook section that earns it. Useful for a quick grading pass.
- [`docs/findings.md`](docs/findings.md) — known limitations and data-quality artifacts (duplicate lyrics in SpotGenTrack, stand-up comedy in the Sad class, the non-Latin-script `clean_text` limitation).

## Repo Structure

```
LyricsMoodPredictor/
├── app/streamlit_app.py          # Streamlit UI
├── src/
│   ├── preprocess.py             # text cleaning, mood labels, gap-zone filter
│   ├── features.py               # TF-IDF vectorizer (classification)
│   ├── classify.py               # split + train + evaluate helpers
│   ├── recommend.py              # MiniLM embeddings + cosine-sim retrieval
│   └── explain.py                # SHAP LinearExplainer wrapper
├── notebooks/
│   ├── 01_eda.ipynb              # EDA + preprocessing experiments
│   ├── 02_modeling.ipynb         # baselines, sweep, best model
│   └── 03_evaluation.ipynb       # error analysis, edge cases, iterations, objective metrics
├── docs/
│   ├── architecture.md           # system architecture + design decisions
│   ├── rubric-mapping.md         # rubric items → evidence locations
│   ├── findings.md               # known limitations + data-quality findings
│   └── design/                   # frontend visual mock + design handoff
│       ├── LyricMood Minimal.html
│       └── DESIGN_HANDOFF.md
├── videos/
│   ├── demo.mp4                  # 3–5 min non-technical demo
│   └── walkthrough.mp4           # 5–10 min technical walkthrough
├── data/processed/               # generated (songs_labeled.csv, gitignored)
├── models/                       # generated (pickles + embeddings, gitignored)
├── results/                      # generated figures + tables
├── README.md                     # this file
├── SETUP.md                      # install + data download
├── ATTRIBUTION.md                # AI-tool + library + dataset usage
└── requirements.txt              # pip dependencies
```
