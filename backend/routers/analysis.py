import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from llm import answer
from models import Audience, Language, Mode, QueryRequest, ResponseBody
from ocr_processor import SUPPORTED_EXTENSIONS, UnsupportedFileError, extract_text

human_router = APIRouter(prefix="/human", tags=["Citizen's Summarizer and Advisor"])
professional_router = APIRouter(prefix="/professional", tags=["Professional Summarizer and Advisor"])

# (router, audience, mode, text endpoint, file endpoint) — paths kept stable for the frontend.
ENDPOINTS = [
    (human_router, Audience.human, Mode.summary, "query-summarizer", "upload-file-human-summarizer"),
    (human_router, Audience.human, Mode.advice, "query-advisor", "upload-file-human-advisor"),
    (professional_router, Audience.professional, Mode.summary, "query-professional-summarizer", "upload-file-professional-summarizer"),
    (professional_router, Audience.professional, Mode.advice, "query-professional-advisor", "upload-file-professional-advisor"),
]
CHUNK_BYTES = 1024 * 1024


async def extract_upload(file: UploadFile) -> tuple[str, list[str]]:
    """Streams the upload to a temp file (capped at MAX_UPLOAD_BYTES) and extracts its text."""
    # Only the extension is taken from the client's filename; the name itself is never used on disk.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type '{suffix}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
    temp_path = Path(temp_name)
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                out.write(chunk)
        # OCR is CPU-bound; keep it off the event loop.
        text, notices = await run_in_threadpool(extract_text, temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return text, notices


def _register(router: APIRouter, audience: Audience, mode: Mode, query_path: str, upload_path: str):
    async def handle_query(data: QueryRequest) -> ResponseBody:
        return await answer(audience, mode, data.query, data.language)

    async def handle_upload(language: Language = Language.eng, file: UploadFile = File(...)) -> ResponseBody:
        text, notices = await extract_upload(file)
        result = await answer(audience, mode, text, language, is_document=True)
        result.notices = notices + result.notices
        return result

    router.add_api_route(f"/{query_path}/", handle_query, methods=["POST"], response_model=ResponseBody, name=query_path)
    router.add_api_route(f"/{upload_path}/", handle_upload, methods=["POST"], response_model=ResponseBody, name=upload_path)


for endpoint in ENDPOINTS:
    _register(*endpoint)
