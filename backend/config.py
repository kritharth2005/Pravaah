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
# Separate from the legacy instructor-large index (chroma_langchain_db); vectors are not interchangeable.
CHROMA_DIR = BASE_DIR / "chroma_db_nomic"
