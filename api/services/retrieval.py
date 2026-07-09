"""
Retrieval client layer. Week 1 only needs ping() for /health; vector search
and payload lookup land in Week 3 behind this same protocol.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../../ATTRIBUTION.md.
"""

from typing import Protocol

from qdrant_client import QdrantClient


class RetrievalClient(Protocol):
    def ping(self) -> bool: ...


class QdrantRetrieval:
    def __init__(self, url: str):
        self._client = QdrantClient(url=url, timeout=2)

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
