"""Indexer tests against an in-memory Qdrant."""

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def tiny_corpus():
    df = pd.DataFrame(
        {
            "name": ["Song A", "Song B", "Song C"],
            "artist": ["Artist 1", "Artist 2", "Artist 3"],
            "mood": ["Hype", "Sad", "Hype"],
            "valence": [0.9, 0.1, 0.8],
            "energy": [0.9, 0.2, 0.7],
            "lyrics": ["[Chorus] loud crowd jumping", "rain empty street", "bass kicking night"],
        }
    )
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(3, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return df, emb


def test_strip_section_headers():
    from scripts.index_corpus import strip_section_headers

    assert strip_section_headers("[Chorus] loud crowd") == " loud crowd"
    assert strip_section_headers(None) == ""


def test_resolve_first_artist():
    from scripts.index_corpus import resolve_first_artist

    amap = {"id1": "Taylor", "id2": "Kendrick"}
    assert resolve_first_artist("['id1', 'id2']", amap) == "Taylor"
    assert resolve_first_artist("[]", amap) == "Unknown"
    assert resolve_first_artist("not-a-list", amap) == "Unknown"


def test_index_corpus_upserts_all_rows(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    assert index_corpus(client, df, emb, "songs") == 3
    assert client.count("songs").count == 3


def test_index_corpus_is_idempotent(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    index_corpus(client, df, emb, "songs")  # run twice
    assert client.count("songs").count == 3  # no duplicates


def test_payload_contents_and_excerpt(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    index_corpus(client, df, emb, "songs")
    point = client.retrieve("songs", ids=[0], with_payload=True)[0]
    assert point.payload["title"] == "Song A"
    assert point.payload["mood"] == "Hype"
    assert "[Chorus]" not in point.payload["lyrics_excerpt"]
    assert "lyrics" not in point.payload  # full lyrics never enter Qdrant


def test_row_embedding_mismatch_raises(tiny_corpus):
    from scripts.index_corpus import ensure_collection, index_corpus

    df, emb = tiny_corpus
    client = QdrantClient(":memory:")
    ensure_collection(client, "songs")
    with pytest.raises(ValueError, match="row count"):
        index_corpus(client, df, emb[:2], "songs")
