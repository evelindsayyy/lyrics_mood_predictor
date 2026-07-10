"""Tests for the eval harness against the tiny fixtures — no artifacts, no mlflow."""

import pandas as pd
import pytest

from tests.conftest import TINY_SONGS, FakeMoodModel


@pytest.fixture
def tiny_df():
    # x4 so the stratified 10% test split has >= 1 row per class (sklearn
    # raises if n_test < n_classes on a 15-row frame)
    songs = TINY_SONGS * 4
    return pd.DataFrame(
        {"lyrics": [t for t, _ in songs], "mood": [m for _, m in songs]}
    )


def test_frozen_test_split_is_deterministic(tiny_df):
    from training.evaluate import frozen_test_split

    a = frozen_test_split(tiny_df)
    b = frozen_test_split(tiny_df)
    assert list(a.index) == list(b.index)
    assert 0 < len(a) < len(tiny_df)


def test_evaluate_predictor_perfect_oracle(tiny_df):
    from training.evaluate import evaluate_predictor

    class Oracle:
        version = "oracle-v0"

        def predict(self, lyrics, explain=True):
            from api.services.model import PredictionResult

            mood = dict(TINY_SONGS)[lyrics]
            return PredictionResult(mood=mood, confidence=1.0, probabilities={mood: 1.0}, explanation=None)

    r = evaluate_predictor(Oracle(), tiny_df)
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["n"] == len(tiny_df)
    assert r["model_version"] == "oracle-v0"


def test_evaluate_predictor_constant_model_and_gate(tiny_df):
    from training.evaluate import evaluate_predictor, passes_quality_gate

    r = evaluate_predictor(FakeMoodModel(mood="Hype"), tiny_df)
    assert r["accuracy"] == pytest.approx(3 / 15)  # 12/60 — same ratio as 3/15
    # constant predictor == majority baseline → does NOT beat it
    assert passes_quality_gate(r, tiny_df["mood"]) is False


def test_evaluate_limit(tiny_df):
    from training.evaluate import evaluate_predictor

    r = evaluate_predictor(FakeMoodModel(), tiny_df, limit=5)
    assert r["n"] == 5


def test_write_report(tmp_path, tiny_df):
    from training.evaluate import evaluate_predictor, write_report

    r = evaluate_predictor(FakeMoodModel(mood="Hype"), tiny_df)
    path = write_report(r, "fake", out_dir=tmp_path)
    text = path.read_text()
    assert "macro_f1" in text and "Hype" in text
    assert path.name == "eval_fake.md"


def test_log_mlflow_noop_without_mlflow(monkeypatch, tiny_df):
    import builtins

    from training.evaluate import evaluate_predictor, log_mlflow

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = evaluate_predictor(FakeMoodModel(), tiny_df)
    log_mlflow(r, "fake", params={"x": 1})  # must not raise
