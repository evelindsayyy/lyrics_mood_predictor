"""Tests for api.services.transformer against the tiny ONNX fixture."""

import numpy as np
import pytest


def _load(tiny_onnx_dir):
    from api.services.transformer import load_transformer

    return load_transformer(tiny_onnx_dir, version="tiny-onnx-v0")


def test_softmax_rows_sum_to_one():
    from api.services.transformer import softmax

    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert not np.isnan(p).any()  # numerically stable at large magnitudes


def test_predict_returns_prediction_result(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("stadium lights bass kicking loud crowd", explain=False)
    assert r.mood in {"Angry", "Calm", "Hype", "Romantic", "Sad"}
    assert 0.0 < r.confidence <= 1.0
    assert pytest.approx(sum(r.probabilities.values()), abs=1e-5) == 1.0
    assert r.explanation is None  # explanation path covered in test_transformer_explain.py
    assert m.version == "tiny-onnx-v0"


def test_predict_is_deterministic(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    a = m.predict("rain empty street coat chair alone")
    b = m.predict("rain empty street coat chair alone")
    assert a.mood == b.mood and a.confidence == b.confidence


def test_predict_proba_batch_shape(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    p = m._predict_proba(["stadium lights", "rain empty street", "tender heart"])
    assert p.shape == (3, 5)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_unknown_words_still_predict(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("zzzz qqqq xxxx")  # all [UNK]
    assert r.mood in {"Angry", "Calm", "Hype", "Romantic", "Sad"}


def test_load_transformer_missing_file(tmp_path):
    from api.services.model import ArtifactError
    from api.services.transformer import load_transformer

    with pytest.raises(ArtifactError) as exc:
        load_transformer(tmp_path, version="v")
    assert "model.onnx" in str(exc.value)
