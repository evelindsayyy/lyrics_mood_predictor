"""
slowapi rate limiting with the project's error envelope.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 — 30 req/min/IP, 429 + Retry-After). See ../ATTRIBUTION.md.

Adaptation note: slowapi's own SlowAPIMiddleware resolves the target route via
`route.endpoint`, but FastAPI 0.139 / Starlette 1.3 register included routers as
`_IncludedRouter` wrappers that expose no top-level `endpoint`. slowapi therefore
resolves every request to a `None` handler and treats it as exempt — enforcing
nothing. `RateLimitMiddleware` below sidesteps route resolution entirely: it
exempts by path, and charges every other request against a single stable per-IP
bucket (`_rate_limited_scope`), matching the spec's "30 req/min/IP".
"""

from collections.abc import Awaitable, Callable, Iterable

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# /health and /metrics are exempt: both are unauthenticated operational
# endpoints (liveness probe, Prometheus scrape) that must never be throttled.
# The brief's `limiter.exempt(metrics_endpoint)` predates this custom
# middleware (which exempts by exact path, not by endpoint), so /metrics is
# added to the exempt path set here instead.
DEFAULT_EXEMPT_PATHS: tuple[str, ...] = ("/health", "/metrics")


def build_limiter(rate_limit: str) -> Limiter:
    return Limiter(key_func=get_remote_address, default_limits=[rate_limit])


def rate_limit_handler(request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={"error": {"code": "rate_limited", "message": f"rate limit exceeded: {exc.detail}"}},
    )
    response.headers["Retry-After"] = "60"
    return response


def _rate_limited_scope() -> None:
    """Sentinel the limiter keys the global per-IP default limit on.

    Never invoked — only its module/qualified name is read by slowapi to build
    the storage key, so all non-exempt paths share one per-IP bucket.
    """
    raise NotImplementedError


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce the limiter's default per-IP limit for every non-exempt request."""

    def __init__(self, app, exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS):
        super().__init__(app)
        self._exempt_paths = frozenset(exempt_paths)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        limiter: Limiter = request.app.state.limiter
        if not limiter.enabled or request.url.path in self._exempt_paths:
            return await call_next(request)
        try:
            limiter._check_request_limit(request, _rate_limited_scope, in_middleware=True)
        except RateLimitExceeded as exc:
            return rate_limit_handler(request, exc)
        return await call_next(request)
