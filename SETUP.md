# Setup

End-to-end instructions for running LyricMood from a fresh clone.

## 1. Python environment

Python 3.10+ is what I've been using. A venv keeps things tidy:

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the dataset

The raw data isn't in the repo — the main file (`spotify_tracks.csv`) is ~256MB, which is over GitHub's limit. Grab SpotGenTrack from one of:

- Kaggle mirror (easier): https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset
- Mendeley original: https://data.mendeley.com/datasets/4m2x4zngny/1

Unzip it at the project root so the relevant files live at:

```
SpotGenTrack/Data Sources/spotify_tracks.csv        # lyrics + valence + energy (notebooks use this)
SpotGenTrack/Data Sources/spotify_artists.csv       # artist_id -> name (Streamlit app uses this)
```

The other CSVs in the bundle (`spotify_albums.csv`, `Features Extracted/`) aren't needed.

## 3. Regenerate the processed dataset

`data/processed/songs_labeled.csv` is derived from the raw file and is ~150MB — also not committed. Run the first notebook to regenerate it:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

This produces:
- `data/processed/songs_labeled.csv` — cleaned + mood-labeled corpus (~76k songs)
- `results/preprocessing_impact.csv` — before/after metrics for the preprocessing experiments
- A few EDA figures in `results/`

## 4. Train the classifier

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

This produces:
- `models/best_classifier.pkl` — logistic regression, C=1.0, L2, class_weight=balanced
- `models/tfidf_vectorizer.pkl` — fitted TF-IDF (unigrams + bigrams, 10k features)

Takes ~2–3 minutes on a laptop CPU (7 LR/NB configs × the 61k-row train split).

## 5. Embed the corpus for retrieval

The Streamlit app's "5 similar songs" panel needs precomputed MiniLM embeddings for the full corpus. Run this one-liner:

```bash
python -c "
import re, pandas as pd
from src.recommend import load_embedding_model, embed_corpus
df = pd.read_csv('data/processed/songs_labeled.csv')
raw = df['lyrics'].map(lambda t: re.sub(r'\[[^\]]*\]', ' ', t) if isinstance(t, str) else '')
embed_corpus(load_embedding_model(), raw.tolist())
"
```

This produces `models/corpus_embeddings.npy` (~112MB, 76k × 384-d float32, L2-normalized). Takes ~8–15 min on CPU; much faster with an Apple Silicon GPU.

## 5b. Export the query embedder (needed for the API's `/v1/search`, `/v1/similar`, `/v1/songs`)

The API embeds incoming queries with an ONNX export of the same `all-MiniLM-L6-v2` model, so query vectors land in the same space as the corpus embeddings above. This is a one-time, torch-required step (run locally, not in the API container):

```bash
python scripts/export_minilm_onnx.py
```

This produces `models/embedder/model.onnx` and `models/embedder/tokenizer.json`, after verifying parity against the original sentence-transformers model on a handful of test sentences.

## 6. Run the app

```bash
streamlit run app/streamlit_app.py
```

First query after cold start takes ~10s while the model, vectorizer, corpus embeddings, and MiniLM all load into memory. Subsequent queries are fast (cached via `@st.cache_resource`).

## 7. (Optional) run the evaluation notebook

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb
```

Produces the confusion matrix, error analysis, edge-case table, and improvement-iteration results. Doesn't feed anything downstream; it's just for reading.

---

## tl;dr quick path

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ... put SpotGenTrack/ at project root ...
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
python -c "import re, pandas as pd; from src.recommend import load_embedding_model, embed_corpus; df=pd.read_csv('data/processed/songs_labeled.csv'); raw=df['lyrics'].map(lambda t: re.sub(r'\[[^\]]*\]',' ',t) if isinstance(t,str) else ''); embed_corpus(load_embedding_model(), raw.tolist())"
python scripts/export_minilm_onnx.py
docker compose up --build   # or: uvicorn api.main:app --reload  &&  streamlit run app/streamlit_app.py
```
