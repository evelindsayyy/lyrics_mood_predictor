"""
App factory. Lifespan loads real artifacts unless fakes are injected
(tests pass model=/retrieval=). Routes are sync `def` so FastAPI runs
them in its thread pool — blocking sklearn/SHAP inference never blocks
the event loop.

AI attribution: implementation by Claude (Anthropic) based on my specification
(factory + state-injection pattern chosen for testability). See ../ATTRIBUTION.md.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import Settings
from api.errors import register_exception_handlers
from api.logging_setup import configure_logging, request_id_middleware
from api.routes import health, predict
from api.services.model import MoodModel, load_baseline
from api.services.retrieval import QdrantRetrieval, RetrievalClient


def create_app(
    settings: Settings | None = None,
    model: MoodModel | None = None,
    retrieval: RetrievalClient | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load real artifacts only for deps that were not injected. Injected
        # fakes (set eagerly below) are left untouched, so tests never touch
        # pickles and the module-level app still defers loading to startup.
        if not hasattr(app.state, "model"):
            app.state.model = load_baseline(cfg)
        if not hasattr(app.state, "retrieval"):
            app.state.retrieval = QdrantRetrieval(cfg.qdrant_url)
        yield

    app = FastAPI(title="LyricMood API", version="1.0", lifespan=lifespan)
    # Injected deps are set on state immediately so they are available even
    # when TestClient is used without a lifespan context (starlette>=1.x only
    # runs lifespan inside `with TestClient(...)`).
    app.state.settings = cfg
    if model is not None:
        app.state.model = model
    if retrieval is not None:
        app.state.retrieval = retrieval
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    return app


app = create_app()
