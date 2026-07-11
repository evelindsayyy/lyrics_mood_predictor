"""
Assemble the self-contained HF Spaces demo bundle (`demo/`, gitignored).

The bundle carries everything the single-container demo needs EXCEPT the full
lyrics: model artifacts (classifier, vectorizer, registry, transformer,
embedder) plus a file-based serverless Qdrant whose payloads hold only
<=300-char excerpts. `data/processed/songs_labeled.csv` is deliberately left
out, so the API's lyrics store is absent in the container and /v1/songs
single-match analysis 503s by design; predict/search/similar/candidates work.

Usage:
    python scripts/build_demo_bundle.py [--out demo]

Refuses (exit 1) if any source artifact is missing. The corpus embeddings come
from the cached models/corpus_embeddings.npy — this script never re-embeds.

AI attribution: implementation by Claude (Anthropic) based on my specification
(bundle contents, excerpt-only constraint, registry-dir rewrite). Reuses the
indexer's functions directly. See ../ATTRIBUTION.md.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# make `from api.x import y` / `from src.x import y` / `from scripts.x import y`
# work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from qdrant_client import QdrantClient

from api.config import Settings
from scripts.index_corpus import (
    ensure_collection,
    index_corpus,
    resolve_first_artist,
    strip_section_headers,
)
from src.recommend import embed_corpus

# Model files/dirs copied verbatim from models/ into demo/models/.
MODEL_FILES = ("best_classifier.pkl", "tfidf_vectorizer.pkl")
MODEL_DIRS = ("transformer", "embedder")
ARTISTS_CSV = Path("SpotGenTrack/Data Sources/spotify_artists.csv")


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_size(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb:.1f} MB"


def _check_sources(settings: Settings) -> list[str]:
    """Return a list of human-readable messages for every missing source."""
    missing: list[str] = []
    for name in MODEL_FILES:
        if not (settings.model_dir / name).exists():
            missing.append(str(settings.model_dir / name))
    for name in MODEL_DIRS:
        if not (settings.model_dir / name).is_dir():
            missing.append(f"{settings.model_dir / name}/ (directory)")
    if not settings.registry_path.exists():
        missing.append(str(settings.registry_path))
    if not settings.labeled_songs_path.exists():
        missing.append(f"{settings.labeled_songs_path} (needed to build the local qdrant)")
    if not Path("models/corpus_embeddings.npy").exists():
        missing.append("models/corpus_embeddings.npy (cached embeddings — refusing to re-embed)")
    if not ARTISTS_CSV.exists():
        missing.append(str(ARTISTS_CSV))
    return missing


def _copy_models(settings: Settings, out_models: Path) -> None:
    """Copy model artifacts and write a registry whose transformer dir points
    inside the bundle (the API reads spec.dir verbatim, so it must be relative
    to the container's working directory, not the source models/ tree)."""
    out_models.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(settings.model_dir / name, out_models / name)
    for name in MODEL_DIRS:
        dst = out_models / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(settings.model_dir / name, dst)

    registry = json.loads(settings.registry_path.read_text(encoding="utf-8"))
    for spec in registry.get("models", {}).values():
        if "dir" in spec:
            # Always the fixed in-container path, never out_models: the
            # Dockerfile COPYs this bundle to demo/ regardless of the local
            # --out used to build it, so baking in --out (e.g. /tmp/demo)
            # would point the registry outside the container's demo/ layout.
            spec["dir"] = (Path("demo/models") / Path(spec["dir"]).name).as_posix()
    (out_models / "registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )


def _build_local_qdrant(settings: Settings, out_qdrant: Path) -> int:
    """Index the full corpus into a file-based serverless qdrant. Payloads hold
    only excerpts (index_corpus enforces this); embeddings come from the cache."""
    if out_qdrant.exists():
        shutil.rmtree(out_qdrant)

    df = pd.read_csv(settings.labeled_songs_path).reset_index(drop=True)
    artists = pd.read_csv(ARTISTS_CSV, usecols=["id", "name"])
    artist_map = dict(zip(artists["id"], artists["name"]))
    df["artist"] = df["artists_id"].map(lambda s: resolve_first_artist(s, artist_map))

    raw = df["lyrics"].map(strip_section_headers)
    # model=None: embed_corpus returns the cached .npy without touching the
    # embedding model. _check_sources already guaranteed the cache exists, so
    # the re-embed branch (which needs a real model) is never reached here.
    embeddings = embed_corpus(None, raw.tolist())
    if embeddings.shape[0] != len(df):
        raise SystemExit(
            "cached embeddings don't match corpus — delete "
            "models/corpus_embeddings.npy and rebuild it before bundling"
        )

    client = QdrantClient(path=str(out_qdrant))
    try:
        ensure_collection(client, settings.qdrant_collection)
        return index_corpus(client, df, embeddings, settings.qdrant_collection)
    finally:
        client.close()


def build(out: Path) -> int:
    settings = Settings()

    missing = _check_sources(settings)
    if missing:
        print("cannot build demo bundle — missing source artifacts:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    out_models = out / "models"
    out_qdrant = out / "qdrant_local"

    print(f"copying model artifacts -> {out_models}")
    _copy_models(settings, out_models)

    print(f"building local qdrant -> {out_qdrant} (excerpt payloads only)")
    n = _build_local_qdrant(settings, out_qdrant)
    print(f"indexed {n} songs into '{settings.qdrant_collection}'")

    total = _dir_size_bytes(out)
    print(
        f"bundle ready at {out}/  "
        f"(models {_fmt_size(_dir_size_bytes(out_models))}, "
        f"qdrant {_fmt_size(_dir_size_bytes(out_qdrant))}, "
        f"total {_fmt_size(total)})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="demo", help="output bundle directory (default: demo)"
    )
    args = parser.parse_args()
    return build(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
