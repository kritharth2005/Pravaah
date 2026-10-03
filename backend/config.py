import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "hermes3:8b")
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", LLM_MODEL)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Ollama's default context window silently truncates the prompt (instructions first),
# so size it for 7 retrieved chunks + instructions + the user's text.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# --- Paths ---
CORPUS_DIR = BASE_DIR / "PDFS"
CHROMA_DIR = BASE_DIR / "chroma_db_nomic"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
UPLOAD_DIR = BASE_DIR / "uploads"

# --- Limits ---
# Requests above this are rejected outright (422).
MAX_REQUEST_CHARS = int(os.getenv("MAX_REQUEST_CHARS", "200000"))
# Text that fits in one prompt within LLM_NUM_CTX alongside the instructions and retrieved
# context. Longer questions are truncated; longer documents are summarized in parts.
MAX_PROMPT_INPUT_CHARS = int(os.getenv("MAX_PROMPT_INPUT_CHARS", "12000"))
# Documents beyond this are truncated before summarizing: each extra part costs one model call.
MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "40000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "30"))
AUDIO_TTL_MINUTES = int(os.getenv("AUDIO_TTL_MINUTES", "60"))
TTS_TIMEOUT_SECONDS = int(os.getenv("TTS_TIMEOUT_SECONDS", "60"))

# --- HTTP ---
# Any localhost port is allowed (Vite falls back to another port when 3000 is busy);
# add deployed frontend origins as a comma-separated list.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
CORS_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
