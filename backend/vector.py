import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents():
    document_loader = PyPDFDirectoryLoader("PDFS")
    return document_loader.load()


def spilt_documents(documents: list[Document]):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return text_spliter.split_documents(documents)


def get_embedding_function():
    device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": device}
    )
    print(f"Embedding model running on: {device}")
    return embeddings


def calculate_chunk_ids(chunks):
    last_pg_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
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
    vector_store = Chroma(
        persist_directory="chroma_langchain_db",
        embedding_function=get_embedding_function(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])

    print("Number of existing documents in VectorSore:", len(existing_ids))

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata.get("id") not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print("Adding new documents:", len(new_chunks))
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def clear_database():
    if os.path.exists("chroma_langchain_db"):
        shutil.rmtree("chroma_langchain_db")
