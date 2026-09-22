"""
sonicverify Backend - FastAPI entry point.

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config
from app.api import analysis, health, numbers
from app.db.database import init_db
from app.exceptions import AppError
from app.services.analysis_service import cleanup_stale_uploads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sonicverify")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    removed = cleanup_stale_uploads()
    if removed:
        logger.info("Removed %d leftover temporary audio file(s)", removed)
    yield


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=(
        "Backend & integration layer for AI-powered voice-cloning impersonation detection. "
        "Results are risk estimates, not proof of fraud."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(numbers.router)
app.include_router(analysis.router)


# ---------------------------------------------------------------------------
# Clean JSON errors: {"success": false, "error": "..."} - never a stack trace
# ---------------------------------------------------------------------------
def _error(status_code: int, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": message, **extra})


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return _error(exc.status_code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p not in ("body", "query", "path", "form")]
        field = ".".join(loc) or "request"
        msg = str(err.get("msg", "Invalid value"))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        if err.get("type") == "missing":
            msg = f"Missing required field: {field}"
        details.append({"field": field, "message": msg})
    first = details[0]["message"] if details else "Invalid request."
    return _error(422, first, details=details)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    logger.exception("Database error on %s %s", request.method, request.url.path)
    return _error(500, "A database error occurred. Please try again.")


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error on %s %s", request.method, request.url.path)
    return _error(500, "An unexpected server error occurred. Please try again.")


@app.get("/", include_in_schema=False)
def root():
    return {"service": config.APP_NAME, "docs": "/docs", "health": "/api/health"}
    
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
