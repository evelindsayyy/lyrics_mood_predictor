"""
Error contract: every error response is {"error": {"code", "message"}}.

AI attribution: implementation by Claude (Anthropic) based on my specification
(envelope shape and code taxonomy from the design spec §5). See ../ATTRIBUTION.md.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        logger.warning("api_error", code=exc.code, status=exc.status_code)
        return _envelope(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        message = f"{loc}: {first.get('msg', 'invalid input')}"
        logger.warning("validation_error", detail=message)
        return _envelope(422, "validation_error", message)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.error("internal_error", exc_info=exc)
        return _envelope(500, "internal_error", "internal server error")
