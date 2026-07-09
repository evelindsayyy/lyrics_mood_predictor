"""Tests for api.services.model."""

import pytest


def test_predict_returns_known_mood(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd")
    assert r.mood in {"Hype", "Romantic", "Calm", "Sad", "Angry"}
    assert 0.0 < r.confidence <= 1.0
    assert pytest.approx(sum(r.probabilities.values()), abs=1e-6) == 1.0


def test_predict_hype_lyrics_lean_hype(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd jumping party")
    assert r.mood == "Hype"


def test_explanation_contains_input_tokens(tiny_model):
    r = tiny_model.predict("stadium lights bass kicking loud crowd", explain=True)
    assert r.explanation is not None
    assert len(r.explanation) <= 10
    tokens = {t for t, _ in r.explanation}
    assert tokens & {"stadium", "lights", "bass", "kicking", "loud", "crowd"}


def test_explain_false_skips_explanation(tiny_model):
    r = tiny_model.predict("stadium lights bass", explain=False)
    assert r.explanation is None


def test_no_vocab_overlap_explanation_is_none_or_empty(tiny_model):
    # words absent from tiny vocabulary → nothing for SHAP to rank
    r = tiny_model.predict("zzzz qqqq xxxx", explain=True)
    assert r.explanation in (None, [])


def test_load_baseline_missing_artifact_fails_fast(tmp_path):
    from api.config import Settings
    from api.services.model import ArtifactError, load_baseline

    s = Settings(model_dir=tmp_path)  # empty dir — no pickles
    with pytest.raises(ArtifactError) as exc:
        load_baseline(s)
    assert "best_classifier.pkl" in str(exc.value)
