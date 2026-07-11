"""
song_id -> full lyrics store for the /v1/songs pipeline.

Full lyrics never enter Qdrant (copyright + size); the indexer's point IDs
are row positions in the processed corpus, so this store is just the lyrics
column held in memory, indexed by position.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.3 — lyrics fetched from the local processed store by song_id).
See ../../ATTRIBUTION.md.
"""

from pathlib import Path

import pandas as pd


class LyricsStore:
    def __init__(self, lyrics: list):
        self._lyrics = list(lyrics)

    def __len__(self) -> int:
        return len(self._lyrics)

    def get(self, song_id: int) -> str | None:
        if not 0 <= song_id < len(self._lyrics):
            return None
        value = self._lyrics[song_id]
        return value if isinstance(value, str) else None

    @classmethod
    def from_csv(cls, path: Path) -> "LyricsStore":
        df = pd.read_csv(path, usecols=["lyrics"])
        return cls(df["lyrics"].tolist())
