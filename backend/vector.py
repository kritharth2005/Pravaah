import time
from pathlib import Path
from ollama import ResponseError
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DIR, CORPUS_DIR, EMBED_MODEL, OLLAMA_BASE_URL

INGEST_BATCH_SIZE = 256
INGEST_MAX_ATTEMPTS = 3


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text is trained with task prefixes; omitting them degrades retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents([f"search_document: {t}" for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(f"search_query: {text}")


def load_documents():
    document_loader = PyPDFDirectoryLoader(str(CORPUS_DIR))
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    return NomicEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_vector_store():
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_function(),
    )


def calculate_chunk_ids(chunks):
    last_pg_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Store the corpus-relative POSIX path so IDs match across Windows and Linux.
        source = Path(chunk.metadata.get("source")).relative_to(CORPUS_DIR).as_posix()
        chunk.metadata["source"] = source
        page = chunk.metadata.get("page")
        current_pg_id = f"{source}:{page}"

        if current_pg_id == last_pg_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_pg_id}:{current_chunk_index}"
        last_pg_id = current_pg_id

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    vector_store = get_vector_store()

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])

    print("Number of existing documents in vector store:", len(existing_ids))

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata.get("id") not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print("Adding new documents:", len(new_chunks))
        # Batch so a single Ollama embed request never carries the whole corpus.
        for start in range(0, len(new_chunks), INGEST_BATCH_SIZE):
            batch = new_chunks[start : start + INGEST_BATCH_SIZE]
            for attempt in range(1, INGEST_MAX_ATTEMPTS + 1):
                try:
                    vector_store.add_documents(batch, ids=[chunk.metadata["id"] for chunk in batch])
                    break
                except ResponseError as e:
                    # Ollama's model runner occasionally drops mid-request; it restarts on the next call.
                    if attempt == INGEST_MAX_ATTEMPTS:
                        raise
                    print(f"  batch at {start} failed (attempt {attempt}): {e}; retrying")
                    time.sleep(5 * attempt)
            print(f"  embedded {start + len(batch)}/{len(new_chunks)}")
    else:
        print("No new documents to add")
