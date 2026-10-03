# Pravaah backend

A FastAPI service that answers legal questions by retrieving from a statute index and generating with a local Ollama model.

```
upload ──► text / OCR ─┐
text ──────────────────┴─► retrieve (Chroma + nomic-embed-text) ─► prompt per audience/mode ─► hermes3:8b
                                                     ─► translate (optional) ─► TTS ─► { text, audio_path, notices }
```

**Questions and documents are handled differently** ([llm.py](llm.py)):

- **Question** (short text): the statutes retrieved for it are summarised, or applied to the situation in advice mode.
- **Document** (any upload, or pasted text of 120 words or more) in summary mode: the document itself is summarised. Retrieved statutes are used only to explain the provisions it cites. Retrieval searches with pieces taken from across the whole document, not just its opening. Documents longer than 12k characters are condensed part by part (one model call per ~10k characters), then summarised together, up to 40k characters.
- In advice mode, a document is analysed as the user's situation.

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com) with both models pulled:

```sh
ollama pull nomic-embed-text
ollama pull hermes3:8b

uv sync
uv run python ingest.py          # builds chroma_db_nomic/ from PDFS/ (~40 min on CPU)
uv run uvicorn app:app --port 8000
```

`ingest.py` is incremental: rerunning it only embeds chunks that are new. `--reset` rebuilds from scratch; stop the server first.

Image uploads and scanned PDFs also need Tesseract and Poppler (`pdftoppm`) on `PATH`. Without them, those uploads return 503 and everything else keeps working.

## API

| Endpoint | Body |
|----------|------|
| `POST /human/query-summarizer/`, `/human/query-advisor/` | JSON `{"query": "...", "language": "eng"}` |
| `POST /professional/query-professional-summarizer/`, `/professional/query-professional-advisor/` | same |
| `POST /human/upload-file-human-summarizer/?language=eng` (and the matching `upload-file-*` routes) | multipart `file`: PDF, PNG, JPG or TXT, max 10 MB |
| `GET /health` | 200 when Ollama is reachable and the index is non-empty, 503 otherwise |

`language` is one of `eng`, `hin`, `kan`, `tam`, `mal`, `tel`. Responses are `{"text", "audio_path", "notices"}`, where `audio_path` is a per-response file under `/static/audio/` that expires after an hour. `notices` lists non-fatal problems, such as a truncated document or translation or audio that failed.

Errors: 413 file too large, 415 unsupported or mislabelled file, 422 invalid input or unreadable document, 503 Ollama or OCR unavailable. Interactive docs are at `/docs`.

## Configuration

Environment variables, also read from `.env`:

| Variable | Default | |
|----------|---------|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `EMBED_MODEL` / `LLM_MODEL` / `TRANSLATE_MODEL` | `nomic-embed-text` / `hermes3:8b` / `LLM_MODEL` | Changing `EMBED_MODEL` requires `ingest.py --reset` |
| `LLM_NUM_CTX` | `8192` | Context window; Ollama truncates prompts that exceed it |
| `LLM_TEMPERATURE` | `0.2` | |
| `MAX_PROMPT_INPUT_CHARS` | `12000` | Longer questions are truncated; longer documents are summarised in parts |
| `MAX_DOCUMENT_CHARS` | `40000` | Documents are truncated beyond this, with a notice; each extra 10k characters costs one model call |
| `MAX_REQUEST_CHARS` | `200000` | Hard limit for pasted text (422 beyond) |
| `MAX_UPLOAD_MB` / `MAX_OCR_PAGES` | `10` / `30` | |
| `CORS_ORIGINS` | *(none)* | Comma-separated extra origins; any `localhost` port is always allowed |
| `LOG_LEVEL` | `INFO` | Logs record timings and sizes, never query text |

## Tests

```sh
uv run pytest
```

The API tests replace the RAG pipeline with a fake, so they need neither Ollama nor the index.

## Docker

Build from this directory. Ollama stays on the host, and the index must already be built:

```sh
docker build -t pravaah-backend .
docker run -p 8000:8000 pravaah-backend
```

The image includes Tesseract and Poppler, and reaches Ollama at `host.docker.internal:11434`.
