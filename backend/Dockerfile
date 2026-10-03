# Pravaah FastAPI backend.
# Ollama (nomic-embed-text + hermes3:8b) runs on the host, not in this container.
# The vector index (chroma_db_nomic/) is built on the host with `python ingest.py` and copied in.
# Build from this directory: docker build -t pravaah-backend .

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

# OCR for images and scanned PDFs
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

RUN useradd --create-home pravaah

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Owned by the app user: Chroma writes to its index, and audio/uploads are created at runtime
COPY --chown=pravaah . ./
RUN mkdir -p static uploads && chown pravaah static uploads
USER pravaah

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
