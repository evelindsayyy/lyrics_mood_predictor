"""Tests for api.config.Settings."""


def test_settings_defaults():
    from api.config import Settings

    s = Settings()
    assert str(s.model_dir) == "models"
    assert s.baseline_classifier == "best_classifier.pkl"
    assert s.baseline_vectorizer == "tfidf_vectorizer.pkl"
    assert str(s.labeled_songs_path) == "data/processed/songs_labeled.csv"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "songs"
    assert s.shap_background_size == 500
    assert s.max_lyrics_chars == 10_000


def test_settings_env_override(monkeypatch):
    from api.config import Settings

    monkeypatch.setenv("LYRICMOOD_QDRANT_URL", "http://qdrant:6333")
    assert Settings().qdrant_url == "http://qdrant:6333"
