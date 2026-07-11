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
from slowapi.errors import RateLimitExceeded

from api.config import Settings
from api.errors import register_exception_handlers
from api.logging_setup import configure_logging, request_id_middleware
from api.metrics import metrics_endpoint, metrics_middleware
from api.ratelimit import RateLimitMiddleware, build_limiter, rate_limit_handler
from api.routes import health, predict, search, songs
from api.services.embedder import Embedder, load_embedder
from api.services.model import ArtifactError, MoodModel, load_baseline
from api.services.registry import load_registry
from api.services.retrieval import QdrantRetrieval, RetrievalClient
from api.services.songs import LyricsStore
from api.services.transformer import load_transformer

logger = structlog.get_logger()


def create_app(
    settings: Settings | None = None,
    models: dict[str, MoodModel] | None = None,
    default: str | None = None,
    retrieval: RetrievalClient | None = None,
    embedder: Embedder | None = None,
    lyrics_store: LyricsStore | None = None,
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
        if not hasattr(app.state, "embedder"):
            if cfg.embedder_dir.exists():
                app.state.embedder = load_embedder(cfg.embedder_dir)
            else:
                logger.info("embedder_unavailable", dir=str(cfg.embedder_dir))
                app.state.embedder = None
        if not hasattr(app.state, "lyrics_store"):
            if cfg.labeled_songs_path.exists():
                app.state.lyrics_store = LyricsStore.from_csv(cfg.labeled_songs_path)
            else:
                logger.info("lyrics_unavailable", path=str(cfg.labeled_songs_path))
                app.state.lyrics_store = None
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
    if embedder is not None:
        app.state.embedder = embedder
    if lyrics_store is not None:
        app.state.lyrics_store = lyrics_store
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    # Fresh Limiter per create_app: default storage_uri is memory://, and
    # storage_from_string("memory://") builds an independent MemoryStorage per
    # instance, so each app has its own counters (tests don't bleed).
    limiter = build_limiter(cfg.rate_limit)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    # RateLimitMiddleware (not slowapi's SlowAPIMiddleware) — see api/ratelimit.py
    # for why: FastAPI 0.139's _IncludedRouter breaks slowapi's route resolution.
    # /health is exempted by path via the middleware's default exempt set.
    app.add_middleware(RateLimitMiddleware)
    # Middleware registration order note: Starlette's add_middleware/`app.middleware`
    # each insert at the front of the stack, so the LAST registered runs OUTERMOST.
    # Final registration order → request_id, rate-limit, metrics, which yields the
    # runtime stack (outer→inner):
    #   metrics → rate-limit → request_id → ExceptionMiddleware → router.
    # metrics is outermost on purpose: it times the full stack and counts every
    # response, including rate-limited 429s (labelled path="unmatched", since a
    # 429 short-circuits before the router sets scope["route"]). Matched requests
    # — including ApiError 400s turned into responses by ExceptionMiddleware — are
    # counted with the route TEMPLATE because the shared scope dict carries
    # scope["route"] back up after call_next.
    app.middleware("http")(metrics_middleware)
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    app.include_router(search.router, prefix="/v1")
    app.include_router(songs.router, prefix="/v1")
    return app


app = create_app()
