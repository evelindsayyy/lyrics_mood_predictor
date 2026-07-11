"""
App factory. Lifespan loads real artifacts unless fakes are injected
(tests pass models=/default=/retrieval=). Routes are sync `def` so FastAPI
runs them in its thread pool — blocking sklearn/SHAP inference never blocks
the event loop.

AI attribution: implementation by Claude (Anthropic) based on my specification
(factory + state-injection pattern chosen for testability). See ../ATTRIBUTION.md.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from api.config import Settings
from api.errors import register_exception_handlers
from api.logging_setup import configure_logging, request_id_middleware
from api.routes import health, predict
from api.services.model import ArtifactError, MoodModel, load_baseline
from api.services.registry import load_registry
from api.services.retrieval import QdrantRetrieval, RetrievalClient
from api.services.transformer import load_transformer

logger = structlog.get_logger()


def create_app(
    settings: Settings | None = None,
    models: dict[str, MoodModel] | None = None,
    default: str | None = None,
    retrieval: RetrievalClient | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load real artifacts only for state that was not injected. Injected
        # fakes (set eagerly below) are left untouched, so tests never touch
        # pickles and the module-level app still defers loading to startup.
        if not hasattr(app.state, "models"):
            reg = load_registry(cfg.registry_path)
            loaded: dict[str, MoodModel] = {}
            for name, spec in reg.models.items():
                if spec.kind == "baseline":
                    loaded[name] = load_baseline(cfg, version=spec.version)
                elif spec.kind == "onnx" and spec.dir is not None and spec.dir.exists():
                    loaded[name] = load_transformer(spec.dir, spec.version)
                else:
                    logger.info("model_unavailable", model=name)
            if reg.default not in loaded:
                raise ArtifactError(f"registry default {reg.default!r} failed to load")
            app.state.models = loaded
            app.state.default_model = reg.default
            app.state.registry_names = set(reg.models)
        if not hasattr(app.state, "retrieval"):
            app.state.retrieval = QdrantRetrieval(cfg.qdrant_url, cfg.qdrant_collection)
        yield

    app = FastAPI(title="LyricMood API", version="1.0", lifespan=lifespan)
    # Injected deps are set on state immediately so they are available even
    # when TestClient is used without a lifespan context (starlette>=1.x only
    # runs lifespan inside `with TestClient(...)`).
    app.state.settings = cfg
    if models is not None:
        app.state.models = dict(models)
        app.state.default_model = default if default is not None else next(iter(models))
        app.state.registry_names = set(models)
    if retrieval is not None:
        app.state.retrieval = retrieval
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    return app


app = create_app()
