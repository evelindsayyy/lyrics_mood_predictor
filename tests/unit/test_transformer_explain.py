"""Tests for transformer SHAP text explanations on the tiny ONNX model."""


def _load(tiny_onnx_dir):
    from api.services.transformer import load_transformer

    return load_transformer(tiny_onnx_dir, version="tiny-onnx-v0")


def test_explanation_returns_token_weights(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    r = m.predict("stadium lights bass kicking loud crowd", explain=True)
    assert r.explanation is not None
    assert 1 <= len(r.explanation) <= 10
    for token, weight in r.explanation:
        assert isinstance(token, str) and token.strip()
        assert isinstance(weight, float)
    # sorted signed descending
    weights = [w for _, w in r.explanation]
    assert weights == sorted(weights, reverse=True)


def test_explain_false_skips(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    assert m.predict("stadium lights", explain=False).explanation is None


def test_explanation_failure_is_non_fatal(tiny_onnx_dir, monkeypatch):
    m = _load(tiny_onnx_dir)
    monkeypatch.setattr(m, "_explain", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    # predict must catch and degrade, not raise
    r = m.predict("stadium lights", explain=True)
    assert r.explanation is None


def test_long_input_is_capped(tiny_onnx_dir):
    m = _load(tiny_onnx_dir)
    long_text = "stadium lights bass " * 200  # ~4000 chars
    r = m.predict(long_text, explain=True)
    # must complete (input capped to explain_max_chars) and stay bounded
    assert r.explanation is None or len(r.explanation) <= 10


def test_explained_class_matches_returned_mood_for_long_input(tiny_onnx_dir, monkeypatch):
    m = _load(tiny_onnx_dir)
    captured = {}
    orig = m._explain

    def spy(text, class_idx):
        captured["class_idx"] = class_idx
        return orig(text, class_idx)

    monkeypatch.setattr(m, "_explain", spy)
    long_text = ("rain empty street coat chair alone " * 30) + "stadium lights bass kicking loud crowd " * 30
    r = m.predict(long_text, explain=True)
    labels = ["Angry", "Calm", "Hype", "Romantic", "Sad"]
    assert labels[captured["class_idx"]] == r.mood
