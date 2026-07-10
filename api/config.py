"""
Runtime settings for the LyricMood API.

All values overridable via environment variables prefixed LYRICMOOD_
(e.g. LYRICMOOD_QDRANT_URL) so docker-compose can rewire service URLs.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose the setting names, defaults, and the env-prefix convention; Claude
wrote the class. See ../ATTRIBUTION.md.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LYRICMOOD_")

    model_dir: Path = Path("models")
    baseline_classifier: str = "best_classifier.pkl"
    baseline_vectorizer: str = "tfidf_vectorizer.pkl"
    labeled_songs_path: Path = Path("data/processed/songs_labeled.csv")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "songs"
    registry_path: Path = Path("models/registry.json")
    shap_background_size: int = 500
    max_lyrics_chars: int = 10_000
