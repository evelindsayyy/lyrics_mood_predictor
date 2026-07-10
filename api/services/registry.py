"""
Model registry — pins which models the API loads and which is the default.

The registry file is committed (models/registry.json). After training the
transformer on Colab and dropping artifacts into models/transformer/, flip
"default" to "transformer" to promote it.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 registry pinning). See ../../ATTRIBUTION.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from api.services.model import ArtifactError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "baseline" | "onnx"
    version: str
    dir: Path | None = None


@dataclass(frozen=True)
class Registry:
    default: str
    models: dict[str, ModelSpec]


def load_registry(path: Path) -> Registry:
    if not path.exists():
        raise ArtifactError(f"model registry missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = {
            name: ModelSpec(
                name=name,
                kind=spec["kind"],
                version=spec["version"],
                dir=Path(spec["dir"]) if "dir" in spec else None,
            )
            for name, spec in raw["models"].items()
        }
        default = raw["default"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"model registry unreadable: {path} ({exc})") from exc
    if default not in models:
        raise ArtifactError(f"model registry default {default!r} not in models: {path}")
    return Registry(default=default, models=models)
