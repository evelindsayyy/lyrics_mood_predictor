"""Tests for api.services.embedder against the tiny embedder fixture."""

import numpy as np
import pytest


def _load(tiny_embedder_dir):
    from api.services.embedder import load_embedder

    return load_embedder(tiny_embedder_dir)


def test_strip_section_headers():
    from api.services.embedder import strip_section_headers

    assert "[Chorus]" not in strip_section_headers("[Chorus] loud crowd")
    assert strip_section_headers(None) == ""


def test_embed_shape_and_normalization(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    out = e.embed(["stadium lights bass", "rain empty street coat chair"])
    assert out.shape == (2, 8)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_is_deterministic(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    a = e.embed(["tender heart kitchen door"])
    b = e.embed(["tender heart kitchen door"])
    assert np.allclose(a, b)


def test_padding_does_not_change_embedding(tiny_embedder_dir):
    # same text alone vs batched with a longer neighbor must embed identically
    e = _load(tiny_embedder_dir)
    alone = e.embed(["stadium lights"])[0]
    batched = e.embed(["stadium lights", "rain empty street coat chair alone tonight"])[0]
    assert np.allclose(alone, batched, atol=1e-5)


def test_empty_text_does_not_crash(tiny_embedder_dir):
    e = _load(tiny_embedder_dir)
    out = e.embed([""])
    assert out.shape == (1, 8)
    assert not np.isnan(out).any()


def test_load_embedder_missing_file(tmp_path):
    from api.services.embedder import load_embedder
    from api.services.model import ArtifactError

    with pytest.raises(ArtifactError) as exc:
        load_embedder(tmp_path)
    assert "model.onnx" in str(exc.value)
