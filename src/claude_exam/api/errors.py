"""One error body for every route: `{"detail": "<message>", "code": "<machine code>"}`.

`detail` is always a string, so Unity's JsonUtility can read it, and `code` lets a client
branch on the reason without matching message text. Validation errors keep FastAPI's
per-field list in `errors`.
"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """An expected failure raised by a service; the handler turns it into an HTTP response."""

    status_code: int = 400
    code: str = "bad_request"
    detail: str = "bad request"

    def __init__(self, detail: str | None = None) -> None:
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


def error_body(detail: str, code: str, **extra: object) -> dict:
    return {"detail": detail, "code": code, **extra}


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(error_body(exc.detail, exc.code), status_code=exc.status_code)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        error_body(detail, f"http_{exc.status_code}"),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    summary = "; ".join(
        f"{'.'.join(str(part) for part in e.get('loc', ()))}: {e.get('msg', '')}"
        for e in errors
    )
    return JSONResponse(
        error_body(summary or "invalid request", "validation_error", errors=errors),
        status_code=422,
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
