"""Indexing: PDF -> pages -> overlapping chunks -> embeddings -> Chroma.

The agent in agent.py reads passages from this index and searches it.
"""

import hashlib
import io
import os
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Configuration (override with environment variables or a .env file) -----

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# "provider:model", e.g. "anthropic:claude-sonnet-5-5" or "openai:gpt-4o-mini"
CHAT_MODEL = os.getenv("CHAT_MODEL", "anthropic:claude-sonnet-5-5")
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def get_embeddings() -> Embeddings:
    """Local sentence-transformers model: free, no API key, downloaded on first use."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm() -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    return init_chat_model(CHAT_MODEL, temperature=0)


def get_vector_store(embeddings: Embeddings, persist_dir: str | None = PERSIST_DIR) -> Chroma:
    """Open (or create) the Chroma collection. persist_dir=None keeps it in memory."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_pdf(data: bytes, name: str) -> list[Document]:
    """One Document per page that has extractable text."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": name, "page": number}))
    return pages


def index_pdf(store: Chroma, data: bytes, name: str) -> int:
    """Split, embed and store one PDF. Returns the number of chunks added.

    Returns 0 if this exact file is already indexed. A different file with the
    same name replaces the old one. Raises ValueError if the PDF has no
    extractable text (e.g. a scan without OCR).
    """
    file_hash = hashlib.sha256(data).hexdigest()[:16]
    if store.get(where={"file_hash": file_hash}, limit=1)["ids"]:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(load_pdf(data, name))
    if not chunks:
        raise ValueError(f"No extractable text in {name} (is it a scanned PDF?)")

    delete_document(store, name)
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
    store.add_documents(chunks, ids=[f"{file_hash}-{i}" for i in range(len(chunks))])
    return len(chunks)


def list_documents(store: Chroma) -> dict[str, int]:
    """Indexed file names mapped to their chunk counts."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    return dict(sorted(Counter(m["source"] for m in metadatas).items()))


def delete_document(store: Chroma, name: str) -> None:
    ids = store.get(where={"source": name}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
