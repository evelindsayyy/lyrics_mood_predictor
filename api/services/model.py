"""
Model service layer — wraps the existing src/ baseline (TF-IDF + LR + SHAP)
behind a MoodModel protocol so routes and tests depend on an interface,
not on artifacts. The fine-tuned transformer (Week 2) will implement the
same protocol.

AI attribution: implementation by Claude (Anthropic) based on my specification
(protocol shape, fail-fast artifact loading, explanation semantics carried
over from app/streamlit_app.py). See ../../ATTRIBUTION.md.
"""

from dataclasses import dataclass
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
import structlog

from api.config import Settings
from src.explain import explain_prediction
from src.preprocess import clean_text

logger = structlog.get_logger()


class ArtifactError(Exception):
    """A required model artifact is missing or unreadable."""


@dataclass(frozen=True)
class PredictionResult:
    mood: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[tuple[str, float]] | None


class MoodModel(Protocol):
    version: str

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult: ...


class BaselineMoodModel:
    """TF-IDF + logistic regression with exact SHAP explanations."""

    def __init__(self, clf, vectorizer, background=None, version: str = "baseline-lr-v1"):
        self._clf = clf
        self._vectorizer = vectorizer
        self._background = background
        self.version = version

    def predict(self, lyrics: str, explain: bool = True) -> PredictionResult:
        cleaned = clean_text(lyrics)
        X = self._vectorizer.transform([cleaned])
        probs = self._clf.predict_proba(X)[0]
        classes = list(self._clf.classes_)
        idx = int(np.argmax(probs))
        explanation = self._explain(cleaned, X) if explain else None
        return PredictionResult(
            mood=str(classes[idx]),
            confidence=float(probs[idx]),
            probabilities={str(c): float(p) for c, p in zip(classes, probs)},
            explanation=explanation,
        )

    def _explain(self, cleaned: str, X) -> list[tuple[str, float]] | None:
        """Top-10 input tokens by |SHAP|; None on any failure (non-fatal per spec)."""
        try:
            exp = explain_prediction(
                self._clf, self._vectorizer, cleaned, top_k=10, background=self._background
            )
            sv = exp["shap_values"]
            fn = exp["feature_names"]
            present = X.nonzero()[1]
            pairs = [(str(fn[i]), float(sv[i])) for i in present]
            top = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:10]
            return sorted(top, key=lambda kv: kv[1], reverse=True)
        except Exception as exc:
            # Never log lyrics content; the exception type is enough to triage.
            logger.warning("explain_failed", error=type(exc).__name__)
            return None


def load_baseline(settings: Settings) -> BaselineMoodModel:
    """Load pickled artifacts; fail fast with the offending path in the message."""
    clf_path = settings.model_dir / settings.baseline_classifier
    vec_path = settings.model_dir / settings.baseline_vectorizer
    for path in (clf_path, vec_path):
        if not path.exists():
            raise ArtifactError(f"model artifact missing: {path}")
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    background = None
    if settings.labeled_songs_path.exists():
        df = pd.read_csv(settings.labeled_songs_path, usecols=["lyrics"])
        rng = np.random.default_rng(42)
        n = min(settings.shap_background_size, len(df))
        bg_idx = rng.choice(len(df), size=n, replace=False)
        background = vec.transform(df["lyrics"].iloc[bg_idx].map(clean_text))

    return BaselineMoodModel(clf=clf, vectorizer=vec, background=background)
