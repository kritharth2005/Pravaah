import asyncio
import logging
import time

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    MAX_DOCUMENT_CHARS,
    MAX_PROMPT_INPUT_CHARS,
    OLLAMA_BASE_URL,
)
from models import Audience, Language, Mode, ResponseBody
from multilingual import localize
from prompts import DOCUMENT_PART_SUMMARY, DOCUMENT_PROMPTS, PROMPTS
from vector import get_vector_store

log = logging.getLogger(__name__)

vector_store = get_vector_store()

model = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
    num_ctx=LLM_NUM_CTX,
)

# Summaries take the closest matches; advice uses MMR so different provisions are covered.
RETRIEVAL_K = {Mode.summary: 7, Mode.advice: 5}
RETRIEVERS = {
    Mode.summary: vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K[Mode.summary]}),
    Mode.advice: vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": RETRIEVAL_K[Mode.advice], "fetch_k": 20}
    ),
}

# Pasted text this long is treated as a document to summarize rather than a question.
DOCUMENT_MIN_WORDS = 120
# The embedding model only sees ~2k tokens, so long inputs are searched piece by piece.
RETRIEVAL_PIECE_CHARS = 2000
MAX_RETRIEVAL_PIECES = 12
MATCHES_PER_PIECE = 3
# Parts for long documents, sized to leave room in LLM_NUM_CTX for the part-summary prompt.
DOCUMENT_PART_CHARS = 10000


def looks_like_document(text: str) -> bool:
    return len(text.split()) >= DOCUMENT_MIN_WORDS


def _split(text: str, size: int, overlap: int) -> list[str]:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)


def _truncate(text: str, limit: int, notices: list[str], what: str) -> str:
    if len(text) <= limit:
        return text
    notices.append(f"The {what} is long, so only its first {limit:,} characters were analysed.")
    return text[:limit]


async def retrieve(text: str, mode: Mode) -> list[Document]:
    if len(text) <= RETRIEVAL_PIECE_CHARS:
        return await RETRIEVERS[mode].ainvoke(text)

    # Search with pieces spread across the whole text, so provisions cited anywhere are found,
    # then keep the closest distinct chunks.
    pieces = _split(text, RETRIEVAL_PIECE_CHARS, 0)
    step = max(1, len(pieces) // MAX_RETRIEVAL_PIECES)
    pieces = pieces[::step][:MAX_RETRIEVAL_PIECES]
    results = await asyncio.gather(
        *(vector_store.asimilarity_search_with_score(piece, k=MATCHES_PER_PIECE) for piece in pieces)
    )
    best: dict[str, tuple[Document, float]] = {}
    for doc, distance in (match for matches in results for match in matches):
        key = doc.metadata.get("id", doc.page_content)
        if key not in best or distance < best[key][1]:
            best[key] = (doc, distance)
    ranked = sorted(best.values(), key=lambda match: match[1])
    return [doc for doc, _ in ranked[: RETRIEVAL_K[mode]]]


async def _generate(template: str, **values) -> str:
    messages = ChatPromptTemplate.from_template(template).format_messages(**values)
    return (await model.ainvoke(messages)).content


def _join(documents: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in documents)


async def summarize_document(audience: Audience, document: str, notices: list[str]) -> tuple[str, int]:
    """Summarizes the user's document itself; returns the summary and the number of parts used."""
    document = _truncate(document, MAX_DOCUMENT_CHARS, notices, "document")
    context = _join(await retrieve(document, Mode.summary))

    if len(document) <= MAX_PROMPT_INPUT_CHARS:
        return await _generate(DOCUMENT_PROMPTS[audience], document=document, context=context), 1

    # Too long for one prompt: condense each part, then summarize the condensed parts together.
    parts = _split(document, DOCUMENT_PART_CHARS, 300)
    part_summaries = []
    for number, part in enumerate(parts, start=1):
        part_summaries.append(await _generate(DOCUMENT_PART_SUMMARY, part=number, total=len(parts), text=part))
    condensed = "The document was too long to include in full, so it is given as summaries of its consecutive parts.\n\n" + "\n\n".join(
        f"[Part {number} of {len(parts)}]\n{summary}" for number, summary in enumerate(part_summaries, start=1)
    )
    return await _generate(DOCUMENT_PROMPTS[audience], document=condensed, context=context), len(parts)


async def answer(
    audience: Audience, mode: Mode, text: str, language: Language, is_document: bool | None = None
) -> ResponseBody:
    """is_document: True for uploads; None detects pasted documents by length."""
    started = time.perf_counter()
    notices: list[str] = []
    if is_document is None:
        is_document = looks_like_document(text)

    if mode is Mode.summary and is_document:
        response, parts = await summarize_document(audience, text, notices)
        kind = f"document parts={parts}"
    else:
        question = _truncate(text, MAX_PROMPT_INPUT_CHARS, notices, "document" if is_document else "question")
        context = _join(await retrieve(question, mode))
        response = await _generate(PROMPTS[(audience, mode)], context=context, question=question)
        kind = "question"
    generated = time.perf_counter()

    translated, audio_path, localize_notices = await localize(response, language)

    # Sizes and timings only: inputs describe users' legal situations.
    log.info(
        "%s/%s %s lang=%s input_chars=%d generate=%.1fs localize=%.1fs",
        audience.value, mode.value, kind, language.value, len(text),
        generated - started, time.perf_counter() - generated,
    )
    return ResponseBody(text=translated, audio_path=audio_path, notices=notices + localize_notices)
