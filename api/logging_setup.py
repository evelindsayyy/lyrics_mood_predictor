"""
Structured JSON logging + request-id middleware. Raw lyrics are never logged;
handlers log lengths and codes only.

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

import logging
import uuid

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:
        # Starlette's ServerErrorMiddleware runs the bare-Exception handler in
        # api/errors.py OUTSIDE user middleware, so by the time it logs, this
        # finally has already cleared contextvars and the header line below
        # never runs. Handle it here while request_id/path are still bound and
        # the header can still be set. The inline envelope MUST stay in sync
        # with api/errors.py's _envelope shape ({"error": {"code", "message"}}).
        logger.error("internal_error", exc_info=exc)
        response = JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "internal server error"}},
        )
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["x-request-id"] = rid
    return response
