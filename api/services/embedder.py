"""
Query-time text embeddings — ONNX MiniLM behind an Embedder protocol.

The corpus matrix in Qdrant was embedded with sentence-transformers
all-MiniLM-L6-v2 (L2-normalized). Query vectors must live in the SAME space,
so this module serves a parity-checked ONNX export of that exact model
(scripts/export_minilm_onnx.py) — masked mean-pooling + L2 normalization
reimplemented in numpy. Serving stays torch-free.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.3 — query embedding must match corpus embedding space).
See ../../ATTRIBUTION.md.
"""

import re
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from api.services.model import ArtifactError

REQUIRED_FILES = ("model.onnx", "tokenizer.json")
_EPS = 1e-9


def strip_section_headers(text) -> str:
    """The corpus embedding convention: raw text minus [Verse]-style headers."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[[^\]]*\]", " ", text)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OnnxEmbedder:
    def __init__(self, session, tokenizer, max_len: int = 256):
        self._session = session
        self._tokenizer = tokenizer
        self._max_len = max_len

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        max_len = min(self._max_len, max(max(len(e.ids) for e in encodings), 1))
        ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            n = min(len(enc.ids), max_len)
            ids[i, :n] = enc.ids[:n]
            mask[i, :n] = enc.attention_mask[:n]
        (token_emb,) = self._session.run(
            ["token_embeddings"], {"input_ids": ids, "attention_mask": mask}
        )
        token_emb = np.asarray(token_emb, dtype=np.float32)
        m = mask[:, :, None].astype(np.float32)
        pooled = (token_emb * m).sum(axis=1) / np.clip(m.sum(axis=1), _EPS, None)
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), _EPS, None)
        return (pooled / norms).astype(np.float32)


def load_embedder(model_dir: Path, max_len: int = 256) -> OnnxEmbedder:
    model_dir = Path(model_dir)
    for name in REQUIRED_FILES:
        if not (model_dir / name).exists():
            raise ArtifactError(f"embedder artifact missing: {model_dir / name}")
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return OnnxEmbedder(session, tokenizer, max_len=max_len)
