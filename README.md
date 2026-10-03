# Pravaah

AI summaries and guidance for Indian legal documents. Pravaah answers questions and analyses uploaded documents using retrieval over 23 central statutes (IPC, BNS, CrPC, CPC, Evidence, Contract, Consumer Protection and others), for two audiences:

- **Citizen portal:** plain-language summaries and advice
- **Legal professional portal:** technical summaries and IRAC-style analysis

Responses can be translated into Hindi, Kannada, Tamil, Telugu or Malayalam and read aloud.

Everything except text-to-speech runs locally. [Ollama](https://ollama.com) serves the embedding model and the LLM. Speech uses Microsoft Edge's online TTS service.

## Layout

| Path | What it is |
|------|------------|
| [backend/](backend/) | FastAPI service: OCR, retrieval (Chroma), generation and translation (Ollama), TTS |
| [frontend/Legal-Summarizer/](frontend/Legal-Summarizer/) | React + Vite app (git submodule) |
| [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) | Architecture review and remediation status |

## Quick start (Windows)

Install [Ollama](https://ollama.com/download), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Node.js](https://nodejs.org), then:

```bat
run.bat          :: sets up whatever is missing, then opens the API and the app
run.bat check    :: only reports what is missing
```

On first run, `run.bat` pulls the models, installs dependencies, and builds the vector index, which takes about 40 minutes on CPU. After that it starts in seconds. The app is at http://localhost:3000. To stop, close the "Pravaah API" and "Pravaah UI" windows.

## Manual start

```sh
# 1. Models
ollama pull nomic-embed-text
ollama pull hermes3:8b

# 2. Frontend source (first clone only)
git submodule update --init

# 3. Backend: build the index once, then serve on :8000
cd backend
uv sync
uv run python ingest.py
uv run uvicorn app:app --port 8000

# 4. Frontend: serve on :3000
cd frontend/Legal-Summarizer
npm install
npm run dev
```

Image uploads and scanned PDFs also need [Tesseract](https://github.com/tesseract-ocr/tesseract) and Poppler on the backend host. The Docker image includes both; see [backend/README.md](backend/README.md).

Answers come from an 8B model and can take a few minutes on CPU, longer when translated. Pravaah gives informational guidance only and is not legal advice.
