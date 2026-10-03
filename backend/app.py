import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ollama import ResponseError

from config import CORS_ORIGIN_REGEX, CORS_ORIGINS, LOG_LEVEL, OLLAMA_BASE_URL, STATIC_DIR
from llm import vector_store
from ocr_processor import ExtractionError, OcrUnavailableError, UnsupportedFileError
from routers.analysis import human_router, professional_router

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pravaah")

OLLAMA_UNAVAILABLE = "The local language model is unavailable. Make sure Ollama is running and the models are pulled."


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if vector_store._collection.count() == 0:
        log.warning("The vector index is empty, so answers will have no legal context. Run `python ingest.py` first.")
    yield


app = FastAPI(
    title="Pravaah Legal AI",
    description="API for summarizing and advising on legal documents.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Registered before CORS so CORS wraps it: an exception handler for Exception would run
# outside CORS, and browsers would report the 500 as a network failure.
@app.middleware("http")
async def catch_unexpected_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred while processing the request."})


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(UnsupportedFileError)
async def unsupported_file(_request: Request, exc: UnsupportedFileError):
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(ExtractionError)
async def extraction_failed(_request: Request, exc: ExtractionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(OcrUnavailableError)
async def ocr_unavailable(_request: Request, exc: OcrUnavailableError):
    log.error("%s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# The Ollama client raises ConnectionError when the server is down and ResponseError
# for missing models or crashed runners.
@app.exception_handler(ConnectionError)
@app.exception_handler(ResponseError)
async def model_unavailable(_request: Request, exc: Exception):
    log.error("Ollama request failed: %s", exc)
    return JSONResponse(status_code=503, content={"detail": OLLAMA_UNAVAILABLE})


@app.get("/")
def read_root():
    return {"message": "Welcome to the Pravaah Legal AI API"}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            ollama_ok = (await client.get(f"{OLLAMA_BASE_URL}/api/version")).is_success
    except httpx.HTTPError:
        ollama_ok = False
    indexed_chunks = vector_store._collection.count()
    status = "ok" if ollama_ok and indexed_chunks else "degraded"
    return JSONResponse(
        status_code=200 if status == "ok" else 503,
        content={"status": status, "ollama": ollama_ok, "indexed_chunks": indexed_chunks},
    )


app.include_router(human_router)
app.include_router(professional_router)
