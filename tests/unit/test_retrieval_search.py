"""QdrantRetrieval.search/find_song against in-memory qdrant with real indexed points."""

import numpy as np
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def seeded():
    """3 songs indexed exactly like scripts/index_corpus.py does."""
    import pandas as pd

    from api.services.retrieval import QdrantRetrieval
    from scripts.index_corpus import ensure_collection, index_corpus

    df = pd.DataFrame(
        {
            "name": ["Midnight Rain", "Stadium Anthem", "Kitchen Door"],
            "artist": ["Ava", "The Crowd", "June"],
            "mood": ["Sad", "Hype", "Romantic"],
            "valence": [0.1, 0.9, 0.7],
            "energy": [0.2, 0.9, 0.4],
            "lyrics": ["rain empty street", "loud crowd jumping", "hand in mine kitchen door"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(3, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")

    r = QdrantRetrieval.__new__(QdrantRetrieval)  # bypass URL ctor, inject memory client
    r._client = client
    r._collection = "songs"
    return r, emb


def test_search_returns_ranked_hits(seeded):
    r, emb = seeded
    hits = r.search(emb[0], limit=3)
    assert len(hits) == 3
    assert hits[0].song_id == 0  # exact vector match ranks first
    assert hits[0].title == "Midnight Rain"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_mood_filter(seeded):
    r, emb = seeded
    hits = r.search(emb[0], limit=3, mood="Hype")
    assert [h.mood for h in hits] == ["Hype"]
    assert hits[0].song_id == 1


def test_search_limit(seeded):
    r, emb = seeded
    assert len(r.search(emb[0], limit=1)) == 1


def test_find_song_by_title(seeded):
    r, _ = seeded
    hits = r.find_song("midnight rain")
    assert hits and hits[0].song_id == 0
    assert hits[0].score > 0.9  # near-exact title match


def test_find_song_with_artist(seeded):
    r, _ = seeded
    hits = r.find_song("kitchen door", artist="June")
    assert hits and hits[0].song_id == 2


def test_find_song_no_match(seeded):
    r, _ = seeded
    assert r.find_song("zzzz qqqq nonexistent") == []
