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
import structlog
from tokenizers import Tokenizer

from api.services.model import ArtifactError, PredictionResult

REQUIRED_FILES = ("model.onnx", "tokenizer.json", "labels.json")

logger = structlog.get_logger()


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class TransformerMoodModel:
    """Fine-tuned transformer via onnxruntime. Implements MoodModel."""

    def __init__(
        self,
        session,
        tokenizer,
        labels: list[str],
        version: str,
        max_len: int = 256,
        explain_max_chars: int = 300,
        explain_max_evals: int = 64,
    ):
        self._session = session
        self._tokenizer = tokenizer
        self._labels = list(labels)
        self._max_len = max_len
        self._explain_max_chars = explain_max_chars
        self._explain_max_evals = explain_max_evals
        self.version = version

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult:
        probs = self._predict_proba([lyrics])[0]
        idx = int(np.argmax(probs))
        explanation = None
        if explain:
            try:
                explanation = self._explain(lyrics, idx)
            except Exception as exc:
                logger.warning("explain_failed", error=type(exc).__name__)
        return PredictionResult(
            mood=self._labels[idx],
            confidence=float(probs[idx]),
            probabilities={l: float(p) for l, p in zip(self._labels, probs)},
            explanation=explanation,
        )

    def _explain(self, lyrics: str, class_idx: int) -> list[tuple[str, float]] | None:
        """Token-level SHAP via the Text masker; capped for latency.

        `class_idx` is the argmax computed once in predict() on the FULL text,
        so the explained class always matches the returned mood even when the
        char cap below would shift the argmax on truncated input.
        """
        import shap  # local import: keeps module import light

        text = lyrics[: self._explain_max_chars]

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
