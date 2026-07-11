"""
TF-IDF features for the lyric classifier.

Keeping this separate from recommend.py on purpose — classification uses TF-IDF
(sparse, interpretable, works well with logistic regression + SHAP) and retrieval
uses MiniLM embeddings (semantic, dense). Mixing them would blur which model
is doing what.

Bigrams are in because a lot of mood signal shows up as short phrases
("break my heart", "turn it up") rather than single words.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose the TF-IDF settings (max_features=20000, ngram_range=(1,2), min_df=3,
sublinear_tf=True after my second-pass tuning) and the function signatures.
Claude wrote the wrappers. See ../ATTRIBUTION.md for the full breakdown.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """TF-IDF vectorizer with unigrams + bigrams by default.

    Defaults landed here after a second-pass sweep (see 03_evaluation.ipynb):
    20k features beats 10k, min_df=3 drops more noise than min_df=2, and
    sublinear_tf=True damps the dominance of very long songs that would
    otherwise monopolize the feature weights.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )


def fit_and_transform(texts, vectorizer: TfidfVectorizer):
    """Fit the vectorizer on texts, return the sparse feature matrix."""
    return vectorizer.fit_transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = "models/tfidf_vectorizer.pkl") -> None:
    """Serialize the fitted vectorizer to disk with joblib."""
    joblib.dump(vectorizer, path)
