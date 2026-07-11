"""Local-path (serverless) qdrant mode — index then serve from the same dir."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def local_dir(tmp_path):
    from qdrant_client import QdrantClient

    from scripts.index_corpus import ensure_collection, index_corpus

    df = pd.DataFrame(
        {
            "name": ["Midnight Rain", "Stadium Anthem"],
            "artist": ["Ava", "The Crowd"],
            "mood": ["Sad", "Hype"],
            "valence": [0.1, 0.9],
            "energy": [0.2, 0.9],
            "lyrics": ["rain empty street", "loud crowd jumping"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(2, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    d = tmp_path / "qdrant_local"
    client = QdrantClient(path=str(d))
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    client.close()
    return d, emb


def test_local_mode_serves_index(local_dir):
    from api.services.retrieval import QdrantRetrieval

    d, emb = local_dir
    r = QdrantRetrieval.local(d)
    assert r.ping() is True
    assert r.count() == 2
    hits = r.search(emb[0], limit=2)
    assert hits[0].title == "Midnight Rain"
    finds = r.find_song("stadium anthem")
    assert finds and finds[0].song_id == 1


def test_settings_qdrant_path(monkeypatch, tmp_path):
    from api.config import Settings

    monkeypatch.setenv("LYRICMOOD_QDRANT_PATH", str(tmp_path / "q"))
    assert str(Settings().qdrant_path) == str(tmp_path / "q")
    monkeypatch.delenv("LYRICMOOD_QDRANT_PATH")
    assert Settings().qdrant_path is None
