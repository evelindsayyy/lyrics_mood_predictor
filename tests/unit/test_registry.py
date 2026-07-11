"""Tests for api.services.registry."""

import json

import pytest


def _write(tmp_path, payload):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(payload))
    return p


GOOD = {
    "default": "baseline",
    "models": {
        "baseline": {"kind": "baseline", "version": "baseline-lr-v1"},
        "transformer": {"kind": "onnx", "version": "distilbert-mood-v1", "dir": "models/transformer"},
    },
}


def test_load_registry_roundtrip(tmp_path):
    from api.services.registry import load_registry

    reg = load_registry(_write(tmp_path, GOOD))
    assert reg.default == "baseline"
    assert set(reg.models) == {"baseline", "transformer"}
    assert reg.models["transformer"].kind == "onnx"
    assert str(reg.models["transformer"].dir) == "models/transformer"
    assert reg.models["baseline"].dir is None


def test_load_registry_missing_file(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    with pytest.raises(ArtifactError) as exc:
        load_registry(tmp_path / "nope.json")
    assert "nope.json" in str(exc.value)


def test_load_registry_unknown_default(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    bad = {"default": "ghost", "models": {"baseline": {"kind": "baseline", "version": "v"}}}
    with pytest.raises(ArtifactError):
        load_registry(_write(tmp_path, bad))


def test_load_registry_malformed_json(tmp_path):
    from api.services.model import ArtifactError
    from api.services.registry import load_registry

    p = tmp_path / "registry.json"
    p.write_text("{not json")
    with pytest.raises(ArtifactError):
        load_registry(p)


def test_committed_registry_is_loadable():
    from api.services.registry import load_registry

    reg = load_registry(__import__("pathlib").Path("models/registry.json"))
    assert reg.default == "transformer"  # promoted 2026-07-10 after beating baseline on the frozen split
    assert "baseline" in reg.models
