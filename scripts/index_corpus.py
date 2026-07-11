"""
Populate the Qdrant `songs` collection from the processed corpus.

Idempotent: point IDs are row positions, so re-running upserts in place.
Full lyrics never enter Qdrant (copyright + size) — only a ~300-char excerpt.

Usage (Qdrant running via docker compose):
    python scripts/index_corpus.py

Serverless local-path mode (single-container demo, no server process):
    python scripts/index_corpus.py --local-path demo/qdrant_local
Local mode takes an exclusive lock — do NOT run the API against the same
path while indexing.

AI attribution: implementation by Claude (Anthropic) based on my specification
(schema from design spec §3.3; artist resolution logic carried over from
app/streamlit_app.py). See ../ATTRIBUTION.md.
"""

import argparse
import ast
import os
import re
import sys

# make `from api.x import y` / `from src.x import y` work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

VECTOR_SIZE = 384
EXCERPT_CHARS = 300


def strip_section_headers(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[[^\]]*\]", "", text)


def resolve_first_artist(ids_str, artist_map: dict) -> str:
    try:
        ids = ast.literal_eval(ids_str)
        return artist_map.get(ids[0], "Unknown") if ids else "Unknown"
    except (ValueError, SyntaxError):
        return "Unknown"


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        name, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    client.create_payload_index(name, "mood", PayloadSchemaType.KEYWORD)
    text_params = TextIndexParams(type="text", tokenizer=TokenizerType.WORD, lowercase=True)
    client.create_payload_index(name, "title", text_params)
    client.create_payload_index(name, "artist", text_params)


def index_corpus(
    client: QdrantClient,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    collection: str,
    batch_size: int = 256,
) -> int:
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"row count mismatch: {len(df)} corpus rows vs {embeddings.shape[0]} embeddings"
        )
    total = 0
    points: list[PointStruct] = []
    for pos, row in enumerate(df.itertuples(index=False)):
        excerpt = strip_section_headers(row.lyrics).strip()[:EXCERPT_CHARS]
        points.append(
            PointStruct(
                id=pos,
                vector=embeddings[pos].tolist(),
                payload={
                    "song_id": pos,
                    "title": row.name,
                    "artist": row.artist,
                    "mood": row.mood,
                    "valence": float(row.valence),
                    "energy": float(row.energy),
                    "lyrics_excerpt": excerpt,
                },
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection, points=points)
            total += len(points)
            points = []
    if points:
        client.upsert(collection, points=points)
        total += len(points)
    return total


def main() -> int:
    from api.config import Settings
    from src.recommend import embed_corpus, load_embedding_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-path",
        default=None,
        help="Index into a file-based serverless qdrant at this path instead of "
        "the URL. Takes an exclusive lock — don't serve the same path while indexing.",
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.labeled_songs_path.exists():
        print(f"missing {settings.labeled_songs_path} — run notebooks/01_eda.ipynb first")
        return 1

    df = pd.read_csv(settings.labeled_songs_path).reset_index(drop=True)
    artists = pd.read_csv("SpotGenTrack/Data Sources/spotify_artists.csv", usecols=["id", "name"])
    artist_map = dict(zip(artists["id"], artists["name"]))
    df["artist"] = df["artists_id"].map(lambda s: resolve_first_artist(s, artist_map))

    raw = df["lyrics"].map(strip_section_headers)
    embeddings = embed_corpus(load_embedding_model(), raw.tolist())  # reuses .npy cache
    if embeddings.shape[0] != len(df):
        print("cached embeddings don't match corpus — delete models/corpus_embeddings.npy and rerun")
        return 1

    if args.local_path is not None:
        client = QdrantClient(path=args.local_path)
        target = f"local-path {args.local_path}"
    else:
        client = QdrantClient(url=settings.qdrant_url)
        target = settings.qdrant_url
    print(f"indexing into {target}")
    ensure_collection(client, settings.qdrant_collection)
    n = index_corpus(client, df, embeddings, settings.qdrant_collection)
    print(f"indexed {n} songs into '{settings.qdrant_collection}' at {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
