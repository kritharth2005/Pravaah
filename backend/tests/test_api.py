"""API tests with the RAG pipeline replaced by a fake, so no Ollama or index is needed."""

from io import BytesIO

import ollama
import pytest
import pytesseract
from fastapi.testclient import TestClient
from PIL import Image

import routers.analysis as analysis
from app import app
from models import ResponseBody

PDF_BYTES = b"%PDF-1.4 not really a pdf"


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    async def fake_answer(audience, mode, query, language, is_document=None):
        recorded.append((audience.value, mode.value, query, language.value, is_document))
        return ResponseBody(text="answer", audio_path="static/audio/x.mp3", notices=["from pipeline"])

    monkeypatch.setattr(analysis, "answer", fake_answer)
    return recorded


@pytest.fixture
def client():
    # No context manager: skips the lifespan's index check.
    return TestClient(app)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/human/query-summarizer/", ("human", "summary")),
        ("/human/query-advisor/", ("human", "advice")),
        ("/professional/query-professional-summarizer/", ("professional", "summary")),
        ("/professional/query-professional-advisor/", ("professional", "advice")),
    ],
)
def test_query_routes_map_to_audience_and_mode(client, calls, path, expected):
    response = client.post(path, json={"query": "  theft  ", "language": "hin"})

    assert response.status_code == 200
    assert response.json() == {"text": "answer", "audio_path": "static/audio/x.mp3", "notices": ["from pipeline"]}
    # Text requests leave document detection to the pipeline.
    assert calls == [(*expected, "theft", "hin", None)]


@pytest.mark.parametrize("body", [{"query": "x", "language": "english"}, {"query": "   "}, {"query": "x" * 200_001}])
def test_invalid_queries_are_rejected_before_the_model_runs(client, calls, body):
    assert client.post("/human/query-summarizer/", json=body).status_code == 422
    assert calls == []


def test_uploads_are_passed_whole_as_documents(client, calls):
    text = "Judgment. " * 5000

    response = client.post(
        "/professional/upload-file-professional-summarizer/?language=tam",
        files={"file": ("judgment.txt", text.encode(), "text/plain")},
    )

    assert response.status_code == 200
    assert calls == [("professional", "summary", text, "tam", True)]


def test_extraction_notices_come_before_pipeline_notices(client, calls, monkeypatch):
    monkeypatch.setattr(analysis, "extract_text", lambda _path: ("scanned text", ["Only the first 30 pages were read."]))

    response = client.post("/human/upload-file-human-advisor/", files={"file": ("a.pdf", PDF_BYTES, "application/pdf")})

    assert response.json()["notices"] == ["Only the first 30 pages were read.", "from pipeline"]


def test_upload_rejects_unsupported_extension(client, calls):
    response = client.post("/human/upload-file-human-summarizer/", files={"file": ("a.docx", b"PK..", "application/octet-stream")})
    assert response.status_code == 415
    assert calls == []


def test_upload_rejects_contents_that_dont_match_extension(client, calls):
    response = client.post("/human/upload-file-human-summarizer/", files={"file": ("a.pdf", b"hello", "application/pdf")})
    assert response.status_code == 415
    assert "not a valid .pdf" in response.json()["detail"]


def test_upload_rejects_oversized_files(client, calls, monkeypatch):
    monkeypatch.setattr(analysis, "MAX_UPLOAD_BYTES", 8)
    response = client.post("/human/upload-file-human-summarizer/", files={"file": ("a.pdf", PDF_BYTES, "application/pdf")})
    assert response.status_code == 413
    assert calls == []


def test_upload_leaves_no_temp_files(client, calls):
    before = set(analysis.UPLOAD_DIR.glob("*"))
    client.post("/human/upload-file-human-summarizer/", files={"file": ("a.txt", b"some text", "text/plain")})
    client.post("/human/upload-file-human-summarizer/", files={"file": ("a.pdf", b"bad", "application/pdf")})
    assert set(analysis.UPLOAD_DIR.glob("*")) == before


def test_empty_document_is_unprocessable(client, calls):
    response = client.post("/human/upload-file-human-advisor/", files={"file": ("a.txt", b"   \n", "text/plain")})
    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize("error", [ConnectionError("Failed to connect to Ollama"), ollama.ResponseError("model not found", 404)])
def test_ollama_failures_return_503(client, monkeypatch, error):
    async def failing_answer(*_args):
        raise error

    monkeypatch.setattr(analysis, "answer", failing_answer)
    response = client.post("/human/query-summarizer/", json={"query": "theft"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_unexpected_errors_return_generic_500_with_cors(monkeypatch):
    async def broken_answer(*_args):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(analysis, "answer", broken_answer)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/human/query-summarizer/", json={"query": "theft"}, headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unknown_origins(client):
    response = client.options(
        "/human/query-summarizer/",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_missing_ocr_engine_returns_503(client, calls, monkeypatch):
    def no_tesseract(*_args, **_kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", no_tesseract)
    response = client.post("/human/upload-file-human-summarizer/", files={"file": ("a.png", png_bytes(), "image/png")})

    assert response.status_code == 503
    assert "OCR is not available" in response.json()["detail"]
    assert calls == []


def test_corrupt_image_is_unprocessable(client, calls):
    response = client.post("/human/upload-file-human-summarizer/", files={"file": ("a.png", b"\x89PNG\r\n\x1a\nbroken", "image/png")})
    assert response.status_code == 422
    assert calls == []


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()
