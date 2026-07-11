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
from api.metrics import metrics_endpoint, metrics_middleware
from api.ratelimit import RateLimitMiddleware, build_limiter
from api.routes import health, predict, search, songs
from api.services.embedder import Embedder, load_embedder
from api.services.model import ArtifactError, MoodModel, load_baseline
from api.services.registry import load_registry
from api.services.retrieval import QdrantRetrieval, RetrievalClient
from api.services.songs import LyricsStore
from api.services.transformer import load_transformer

logger = structlog.get_logger()


def _lyrics_store_consistent(retrieval: RetrievalClient, store: LyricsStore) -> bool:
    """True if the lyrics store and the Qdrant collection agree on row count.

    The song_id contract is row-position-at-index-time, so a regenerated CSV
    served against a stale index silently returns the wrong lyrics. This guards
    that skew. On any count() failure the collection size is unknown, so we
    return True (don't disable the store on transient errors).

    Calls retrieval.count() at most once and logs the lyrics_index_skew
    warning itself on mismatch — callers only need the returned bool.
    """
    try:
        collection_points = retrieval.count()
    except Exception:
        logger.debug("lyrics_index_count_unavailable")
        return True
    store_rows = len(store)
    consistent = store_rows == collection_points
    if not consistent:
        logger.warning("lyrics_index_skew", store_rows=store_rows, collection_points=collection_points)
    return consistent


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
            if cfg.qdrant_path is not None:
                app.state.retrieval = QdrantRetrieval.local(cfg.qdrant_path, cfg.qdrant_collection)
            else:
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
        # Skew guard, real path only (neither retrieval nor lyrics_store injected):
        # if the in-memory lyrics store and the Qdrant collection disagree on row
        # count, the song_id -> lyrics mapping is stale. Disable the store so the
        # /v1/songs route degrades to its existing 503 rather than serving wrong
        # lyrics. Skipped silently when the collection count is unknown.
        if (
            retrieval is None
            and lyrics_store is None
            and app.state.lyrics_store is not None
            and app.state.retrieval.ping()
            and not _lyrics_store_consistent(app.state.retrieval, app.state.lyrics_store)
        ):
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
    # No app-level RateLimitExceeded handler: RateLimitMiddleware catches the
    # exception and builds the 429 itself (see api/ratelimit.py), so an
    # app.add_exception_handler(RateLimitExceeded, ...) would be unreachable.
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
