import asyncio
import logging
import re
import time
import uuid

import edge_tts
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import (
    AUDIO_DIR,
    AUDIO_TTL_MINUTES,
    BASE_DIR,
    LLM_NUM_CTX,
    OLLAMA_BASE_URL,
    TRANSLATE_MODEL,
    TTS_TIMEOUT_SECONDS,
)
from models import Language

log = logging.getLogger(__name__)

# Language code -> (name used in the translation prompt, edge-tts voice)
LANGUAGES = {
    Language.eng: ("english", "en-US-AriaNeural"),
    Language.hin: ("hindi", "hi-IN-MadhurNeural"),
    Language.kan: ("kannada", "kn-IN-SapnaNeural"),
    Language.tam: ("tamil", "ta-IN-PallaviNeural"),
    Language.mal: ("malayalam", "ml-IN-MidhunNeural"),
    Language.tel: ("telugu", "te-IN-MohanNeural"),
}

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert language translator. Your task is to translate the user's text into the specified language. Return only the translated text and nothing else."),
        ("human", "Translate the following text to {lang}:\n\n{text_to_translate}"),
    ]
)

translator = TRANSLATE_PROMPT | ChatOllama(
    model=TRANSLATE_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=LLM_NUM_CTX,
) | StrOutputParser()


async def translate(text: str, lang: str) -> str:
    return await translator.ainvoke({"text_to_translate": text, "lang": lang})


def speakable(text: str) -> str:
    """Strips Markdown so the TTS voice doesn't read out symbols."""
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"[*_`]+", "", text)


def _remove_expired_audio():
    cutoff = time.time() - AUDIO_TTL_MINUTES * 60
    for path in AUDIO_DIR.glob("*.mp3"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


async def synthesize(text: str, voice: str) -> str:
    """Writes the speech to its own file and returns its path relative to the API root."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _remove_expired_audio()
    # Unguessable per-request name: responses are private to the requester.
    path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(speakable(text), voice, rate="+10%")
    await asyncio.wait_for(communicate.save(str(path)), TTS_TIMEOUT_SECONDS)
    return path.relative_to(BASE_DIR).as_posix()


async def localize(text: str, language: Language) -> tuple[str, str | None, list[str]]:
    """Translates and voices a response. Failures degrade to English text / no audio."""
    notices = []
    lang_name, voice = LANGUAGES[language]

    if language != Language.eng:
        try:
            text = await translate(text, lang_name)
        except Exception:
            log.exception("Translation to %s failed", lang_name)
            notices.append(f"Translation to {lang_name.title()} failed, so the response is shown in English.")
            voice = LANGUAGES[Language.eng][1]

    try:
        audio_path = await synthesize(text, voice)
    except Exception:
        log.exception("Text-to-speech failed")
        notices.append("Audio could not be generated for this response.")
        audio_path = None

    return text, audio_path, notices
