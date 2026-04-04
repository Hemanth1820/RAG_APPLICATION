from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from typing import Optional

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context var — set by main.py before each graph invocation so that
# search_docs can read the active thread without changing its LLM-visible signature.
# ---------------------------------------------------------------------------
active_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_thread_id", default="default"
)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_store: Optional[Chroma] = None
_lock: asyncio.Lock = asyncio.Lock()


def get_embeddings() -> BedrockEmbeddings:
    """Returns a configured BedrockEmbeddings instance."""
    return BedrockEmbeddings(
        model_id=settings.bedrock_embed_model_id,
        region_name=settings.aws_region,
    )


def init_store() -> None:
    """
    Called once at FastAPI lifespan startup.
    Creates or re-opens the ChromaDB collection from disk.
    """
    global _store
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    _store = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )
    count = _store._collection.count()
    logger.info(
        "ChromaDB initialised — collection '%s', %d existing vectors",
        settings.chroma_collection_name,
        count,
    )


async def add_documents_to_store(documents: list[Document], thread_id: str) -> int:
    """
    Thread-safe document ingestion.
    Tags every chunk with thread_id so searches can be filtered per-thread.
    Returns total vector count in the collection after insertion.
    """
    if _store is None:
        raise RuntimeError("Vector store not initialised. Call init_store() first.")

    for doc in documents:
        doc.metadata["thread_id"] = thread_id

    loop = asyncio.get_event_loop()
    async with _lock:
        await loop.run_in_executor(None, lambda: _store.add_documents(documents))

    total = _store._collection.count()
    logger.info(
        "Added %d chunks for thread '%s' — total vectors: %d",
        len(documents), thread_id, total,
    )
    return total


async def similarity_search(query: str, k: int = 4, thread_id: str = "default") -> list[Document]:
    """
    Async similarity search filtered to the given thread_id.
    Raises ValueError if the store has no documents for this thread.
    """
    if not is_store_ready_for_thread(thread_id):
        raise ValueError(
            f"No documents found for thread '{thread_id}'. "
            "Please upload a PDF document first."
        )
    return await _store.asimilarity_search(query, k=k, filter={"thread_id": thread_id})


def is_store_ready_for_thread(thread_id: str) -> bool:
    """True when the store has at least one document tagged with this thread_id."""
    if _store is None:
        return False
    results = _store._collection.get(where={"thread_id": thread_id}, limit=1)
    return len(results["ids"]) > 0


def is_store_ready() -> bool:
    """Global readiness check — used by /health endpoint only."""
    return _store is not None and _store._collection.count() > 0
