"""
Retrieval client layer. Week 1 needed only ping() for /health; Week 3 adds
vector search and payload lookup behind this same protocol.

`search`/`find_song` let ALL qdrant exceptions propagate so routes can 503;
`ping()` stays the graceful (never-raises) one used by /health.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../../ATTRIBUTION.md.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client import models as qm

# find_song fallback (see find_song docstring): bound the client-side scan and
# require a minimum fuzzy-title similarity so a no-match query returns [].
_FALLBACK_SCAN_LIMIT = 10_000
_FALLBACK_MIN_RATIO = 0.5


def _title_ratio(query: str, title: str) -> float:
    return SequenceMatcher(None, query.lower(), title.lower()).ratio()


@dataclass(frozen=True)
class SongHit:
    song_id: int
    title: str
    artist: str
    mood: str
    score: float


class RetrievalClient(Protocol):
    def ping(self) -> bool: ...

    def search(self, vector, limit: int = 10, mood: str | None = None) -> list[SongHit]: ...

    def find_song(self, title: str, artist: str | None = None, limit: int = 5) -> list[SongHit]: ...


def _hit_from_payload(payload: dict, score: float) -> SongHit:
    return SongHit(
        song_id=int(payload["song_id"]),
        title=str(payload["title"]),
        artist=str(payload["artist"]),
        mood=str(payload["mood"]),
        score=float(score),
    )


class QdrantRetrieval:
    def __init__(self, url: str, collection: str = "songs"):
        self._client = QdrantClient(url=url, timeout=5)
        self._collection = collection

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def search(self, vector, limit: int = 10, mood: str | None = None) -> list[SongHit]:
        flt = None
        if mood is not None:
            flt = qm.Filter(must=[qm.FieldCondition(key="mood", match=qm.MatchValue(value=mood))])
        res = self._client.query_points(
            self._collection,
            query=list(map(float, vector)),
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [_hit_from_payload(p.payload, p.score) for p in res.points]

    def find_song(self, title: str, artist: str | None = None, limit: int = 5) -> list[SongHit]:
        """Fuzzy title lookup, ranked client-side by difflib ratio (not qdrant order).

        Primary path: a full-text ``MatchText`` coarse filter on title (+ artist),
        which the server's lowercase text index makes case-insensitive. In-memory
        / index-less qdrant ignores that index and matches ``MatchText``
        case-sensitively, so a lowercased query finds nothing there; when the
        coarse filter returns no candidates we fall back to a bounded, case-
        insensitive client-side fuzzy scan (min-ratio gated so a genuine
        no-match still returns ``[]``).
        """
        must = [qm.FieldCondition(key="title", match=qm.MatchText(text=title))]
        if artist:
            must.append(qm.FieldCondition(key="artist", match=qm.MatchText(text=artist)))
        points, _ = self._client.scroll(
            self._collection,
            scroll_filter=qm.Filter(must=must),
            limit=max(limit * 4, 20),
            with_payload=True,
        )
        if points:
            scored = [
                _hit_from_payload(p.payload, _title_ratio(title, str(p.payload["title"])))
                for p in points
            ]
        else:
            scored = self._fuzzy_scan(title, artist)
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def _fuzzy_scan(self, title: str, artist: str | None) -> list[SongHit]:
        """Bounded client-side fuzzy fallback for the find_song coarse filter."""
        cand, _ = self._client.scroll(
            self._collection, limit=_FALLBACK_SCAN_LIMIT, with_payload=True
        )
        artist_q = artist.lower() if artist else None
        hits: list[SongHit] = []
        for p in cand:
            if artist_q and artist_q not in str(p.payload["artist"]).lower():
                continue
            ratio = _title_ratio(title, str(p.payload["title"]))
            if ratio >= _FALLBACK_MIN_RATIO:
                hits.append(_hit_from_payload(p.payload, ratio))
        return hits
