"""
Prometheus metrics: request counts + latency histograms per route template.

Route TEMPLATES (e.g. /v1/predict) keep label cardinality bounded — raw query
strings never become label values, and unmatched requests (404s, or requests
short-circuited before routing such as rate-limited 429s) collapse to a single
"unmatched" label instead of leaking arbitrary scanned paths.

Deriving the template — adaptation note: the brief specified
`request.scope["route"].path`, but FastAPI 0.139 mounts prefixed routers as
`_IncludedRouter` sub-mounts (the same wrapper that broke slowapi's route
resolution — see api/ratelimit.py). At the outermost middleware the matched
leaf route's `.path` is therefore the UN-prefixed "/predict", not "/v1/predict".
`request.scope["path"]` does carry the full prefixed path, and because this API
declares NO path parameters (every route uses query params or a request body),
the request path is identical to the route template. So the template is taken
from `scope["path"]`, guarded by a matched `scope["route"]` so unmatched paths
(404 scans) can't inflate label cardinality.

The /metrics scrape endpoint excludes itself from instrumentation so scrapes
don't inflate their own counters.

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.1 metrics endpoint). See ../ATTRIBUTION.md.
"""

import time

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "lyricmood_requests_total", "HTTP requests", ["path", "method", "status"]
)
REQUEST_LATENCY = Histogram("lyricmood_request_seconds", "Request latency", ["path"])

_METRICS_PATH = "/metrics"


def _label_path(request: Request) -> str:
    """Full route template for a matched request, else "unmatched"."""
    if request.scope.get("route") is None:
        return "unmatched"
    return request.scope.get("path", "unmatched")


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = _label_path(request)
    if path != _METRICS_PATH:
        REQUEST_COUNT.labels(
            path=path, method=request.method, status=str(response.status_code)
        ).inc()
        REQUEST_LATENCY.labels(path=path).observe(time.perf_counter() - start)
    return response


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
