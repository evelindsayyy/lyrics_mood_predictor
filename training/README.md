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
       !git clone https://github.com/evelindsayyy/lyrics_mood_predictor.git
       %cd lyrics_mood_predictor
       !pip install -q -r training/requirements-train.txt
       !python training/finetune_distilbert.py \
           --data /content/drive/MyDrive/songs_labeled.csv \
           --out /content/drive/MyDrive/lyricmood_transformer

   If the repo is private, cloning needs a token — simpler alternative: skip
   the clone, upload `training/finetune_distilbert.py` and `src/classify.py`
   to Colab's file panel preserving the `training/`+`src/` layout, and run
   the same command from `/content`.

4. When it finishes, download the four files from
   `Drive/lyricmood_transformer/` (`model.onnx`, `tokenizer.json`,
   `labels.json`, `metrics.json`) into `models/transformer/` locally.

## 3. Evaluate + promote

    python -m training.evaluate --model transformer            # frozen test split, writes results/eval_transformer.md
    python -m training.evaluate --model baseline               # regenerate the comparison row

If the transformer clears the gate and beats the baseline's macro F1:
edit `models/registry.json` → `"default": "transformer"`, restart the API
(`docker compose up --build`), and update the README comparison table from
the two eval reports.

MLflow runs land in `./mlruns` — `mlflow ui` to browse.
