"""Builds the statute vector index from PDFS/ using the configured Ollama embedding model.

    python ingest.py           # add chunks that aren't indexed yet
    python ingest.py --reset   # delete the index and rebuild it from scratch

Stop the API server before --reset: it holds the index open.
"""

import argparse
import shutil

from config import CHROMA_DIR, CORPUS_DIR, EMBED_MODEL
from vector import add_to_chroma, load_documents, split_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the existing index before ingesting")
    args = parser.parse_args()

    if args.reset and CHROMA_DIR.exists():
        print(f"Deleting index at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f"Loading PDFs from {CORPUS_DIR} (embedding model: {EMBED_MODEL})")
    chunks = split_documents(load_documents())
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
