"""Unit tests for ingestion IDs, localization fallbacks, and audio file handling."""

import asyncio
import os
import time

import pytest
from langchain_core.documents import Document

import multilingual
from config import CORPUS_DIR
from models import Language
from vector import calculate_chunk_ids


def test_chunk_ids_are_corpus_relative_and_platform_independent():
    source = str(CORPUS_DIR / "BNS.pdf")
    chunks = [
        Document(page_content="a", metadata={"source": source, "page": 0}),
        Document(page_content="b", metadata={"source": source, "page": 0}),
        Document(page_content="c", metadata={"source": source, "page": 1}),
    ]

    ids = [chunk.metadata["id"] for chunk in calculate_chunk_ids(chunks)]

    assert ids == ["BNS.pdf:0:0", "BNS.pdf:0:1", "BNS.pdf:1:0"]
    assert chunks[0].metadata["source"] == "BNS.pdf"


def test_speakable_strips_markdown():
    text = "### 1. Heading\n* **Law Name:** Act, 2019\n- item `code`"
    assert multilingual.speakable(text) == "1. Heading\nLaw Name: Act, 2019\nitem code"


def test_localize_translates_and_voices_in_the_target_language(monkeypatch):
    async def fake_translate(text, lang):
        return f"[{lang}] {text}"

    voices = []

    async def fake_synthesize(text, voice):
        voices.append(voice)
        return "static/audio/a.mp3"

    monkeypatch.setattr(multilingual, "translate", fake_translate)
    monkeypatch.setattr(multilingual, "synthesize", fake_synthesize)

    assert asyncio.run(multilingual.localize("hello", Language.tam)) == ("[tamil] hello", "static/audio/a.mp3", [])
    assert voices == ["ta-IN-PallaviNeural"]


def test_localize_degrades_to_english_text_without_audio(monkeypatch):
    async def failing(*_args):
        raise RuntimeError("down")

    monkeypatch.setattr(multilingual, "translate", failing)
    monkeypatch.setattr(multilingual, "synthesize", failing)

    text, audio_path, notices = asyncio.run(multilingual.localize("hello", Language.hin))

    assert (text, audio_path) == ("hello", None)
    assert notices == [
        "Translation to Hindi failed, so the response is shown in English.",
        "Audio could not be generated for this response.",
    ]


def test_each_response_gets_its_own_audio_file_and_old_files_expire(monkeypatch, tmp_path):
    class FakeCommunicate:
        def __init__(self, text, voice, rate):
            self.text = text

        async def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text)

    monkeypatch.setattr(multilingual, "AUDIO_DIR", tmp_path / "static" / "audio")
    monkeypatch.setattr(multilingual, "BASE_DIR", tmp_path)
    monkeypatch.setattr(multilingual.edge_tts, "Communicate", FakeCommunicate)

    first = asyncio.run(multilingual.synthesize("one", "voice"))
    stale = tmp_path / first
    old = time.time() - (multilingual.AUDIO_TTL_MINUTES + 1) * 60
    os.utime(stale, (old, old))
    second = asyncio.run(multilingual.synthesize("two", "voice"))

    assert first != second
    assert second.startswith("static/audio/") and second.endswith(".mp3")
    assert not stale.exists()
    assert (tmp_path / second).read_text(encoding="utf-8") == "two"


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_a_voice(language):
    assert language in multilingual.LANGUAGES
