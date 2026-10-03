"""Pipeline tests with a fake model and vector store: which prompt and path each input takes."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import llm
from models import Audience, Language, Mode

WORDS_200 = " ".join(f"word{i}" for i in range(200))


class FakeModel:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        return SimpleNamespace(content=f"part summary {len(self.prompts)}" if "PART " in prompt else "final answer")


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def ainvoke(self, query):
        self.queries.append(query)
        return [Document(page_content="retrieved statute", metadata={"id": "BNS.pdf:1:0"})]


class FakeStore:
    """Returns the same three chunks for every piece, at distances that depend on the piece."""

    def __init__(self):
        self.searches = []

    async def asimilarity_search_with_score(self, piece, k):
        self.searches.append(piece)
        offset = len(self.searches) / 100
        return [
            (Document(page_content=f"chunk {name}", metadata={"id": name}), distance + offset)
            for name, distance in [("a", 0.5), ("b", 0.2), ("c", 0.9)]
        ][:k]


@pytest.fixture
def fakes(monkeypatch):
    model, retriever, store = FakeModel(), FakeRetriever(), FakeStore()
    monkeypatch.setattr(llm, "model", model)
    monkeypatch.setattr(llm, "RETRIEVERS", {Mode.summary: retriever, Mode.advice: retriever})
    monkeypatch.setattr(llm, "vector_store", store)

    async def no_localize(text, language):
        return text, None, []

    monkeypatch.setattr(llm, "localize", no_localize)
    return SimpleNamespace(model=model, retriever=retriever, store=store)


def run(*args, **kwargs):
    return asyncio.run(llm.answer(*args, **kwargs))


def test_short_question_summarizes_the_law_not_a_document(fakes):
    result = run(Audience.human, Mode.summary, "What is the punishment for theft?", Language.eng)

    assert result.text == "final answer"
    assert fakes.retriever.queries == ["What is the punishment for theft?"]
    [prompt] = fakes.model.prompts
    assert "QUESTION: What is the punishment for theft?" in prompt
    assert "DOCUMENT" not in prompt


@pytest.mark.parametrize("audience, heading", [(Audience.human, "### What this document is"), (Audience.professional, "**Nature of Document:**")])
def test_pasted_document_is_summarized_itself(fakes, audience, heading):
    run(audience, Mode.summary, WORDS_200, Language.eng)

    [prompt] = fakes.model.prompts
    assert heading in prompt
    assert WORDS_200 in prompt  # the document itself, not just retrieved statutes
    assert "retrieved statute" in prompt


def test_uploads_are_documents_even_when_short(fakes):
    run(Audience.human, Mode.summary, "Notice to vacate the shop within seven days.", Language.eng, is_document=True)

    [prompt] = fakes.model.prompts
    assert "DOCUMENT:\nNotice to vacate the shop within seven days." in prompt


def test_long_document_is_summarized_in_parts_then_combined(fakes, monkeypatch):
    monkeypatch.setattr(llm, "MAX_PROMPT_INPUT_CHARS", 3000)
    monkeypatch.setattr(llm, "DOCUMENT_PART_CHARS", 2000)
    document = "\n\n".join(f"Paragraph {i}. " + "facts " * 60 for i in range(20))

    result = run(Audience.professional, Mode.summary, document, Language.eng)

    *part_prompts, final_prompt = fakes.model.prompts
    assert len(part_prompts) >= 3
    assert all(f"OF {len(part_prompts)}:" in prompt for prompt in part_prompts)
    assert "Paragraph 0." in part_prompts[0] and "Paragraph 19." in part_prompts[-1]
    assert f"[Part {len(part_prompts)} of {len(part_prompts)}]\npart summary {len(part_prompts)}" in final_prompt
    assert "Paragraph 19." not in final_prompt
    assert result.notices == []


def test_document_beyond_the_cap_is_truncated_with_a_notice(fakes, monkeypatch):
    monkeypatch.setattr(llm, "MAX_DOCUMENT_CHARS", 1000)

    result = run(Audience.human, Mode.summary, WORDS_200 * 3, Language.eng)

    assert result.notices == ["The document is long, so only its first 1,000 characters were analysed."]
    assert WORDS_200[:1000] in fakes.model.prompts[-1]


def test_advice_on_a_long_document_keeps_the_advisor_prompt(fakes, monkeypatch):
    monkeypatch.setattr(llm, "MAX_PROMPT_INPUT_CHARS", 500)

    result = run(Audience.human, Mode.advice, WORDS_200, Language.eng, is_document=True)

    [prompt] = fakes.model.prompts
    assert "### 1. What kind of case is this?" in prompt
    assert result.notices == ["The document is long, so only its first 500 characters were analysed."]


def test_long_inputs_are_retrieved_piece_by_piece_and_deduplicated(fakes, monkeypatch):
    monkeypatch.setattr(llm, "RETRIEVAL_PIECE_CHARS", 300)
    text = "\n\n".join(f"Section {i} " + "x " * 120 for i in range(40))

    documents = asyncio.run(llm.retrieve(text, Mode.advice))

    assert fakes.retriever.queries == []
    assert 1 < len(fakes.store.searches) <= llm.MAX_RETRIEVAL_PIECES
    assert "Section 0 " in fakes.store.searches[0]  # pieces span the whole text, start to end
    assert any("Section 3" in piece for piece in fakes.store.searches[-2:])
    # Same three chunks from every piece: each kept once, closest first, cut to k.
    assert [doc.page_content for doc in documents] == ["chunk b", "chunk a", "chunk c"]
