from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app import store as vector_store
from app.config import settings
from app.engine import build_graph, make_initial_state
from app.store import active_thread_id, is_store_ready_for_thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level graph + checkpointer (initialised in lifespan)
# ---------------------------------------------------------------------------
_graph: Any = None
_checkpointer: Any = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _checkpointer

    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)

    # Initialise ChromaDB
    vector_store.init_store()

    # Determine whether AsyncSqliteSaver is available
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        _has_async_sqlite = True
    except ImportError:
        _has_async_sqlite = False

    if _has_async_sqlite:
        db_path = settings.checkpoint_db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # AsyncSqliteSaver must stay open for the full app lifetime — use it as
        # an async context manager that spans the entire lifespan yield.
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            _checkpointer = checkpointer
            _graph = build_graph(_checkpointer)
            logger.info("AsyncSqliteSaver initialised at %s", db_path)
            logger.info("MARA LangGraph compiled and ready")
            yield
        logger.info("Shutdown complete")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        _graph = build_graph(_checkpointer)
        logger.warning("AsyncSqliteSaver unavailable — using MemorySaver (no persistence)")
        logger.info("MARA LangGraph compiled and ready")
        yield
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MARA — Multi-Agent RAG Architect",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int
    total_vectors: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(default="default")


class StepSummary(BaseModel):
    node: str
    output_summary: str


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    retry_count: int
    review_passed: bool
    steps: list[StepSummary]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "store_ready": vector_store.is_store_ready(),
        "model": settings.bedrock_model_id,
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a PDF and index it into ChromaDB",
)
async def upload_pdf(
    file: UploadFile = File(...),
    thread_id: str = Form(default="default"),
) -> UploadResponse:
    """
    Accepts a PDF, splits it with RecursiveCharacterTextSplitter,
    embeds via BedrockEmbeddings, and stores in ChromaDB tagged to thread_id.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    tmp_path: str | None = None
    try:
        # Write to a temp file — PyPDFLoader requires a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF contained no extractable text.",
            )

        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        total_vectors = await vector_store.add_documents_to_store(chunks, thread_id)

        return UploadResponse(
            message=f"Successfully indexed '{file.filename}'",
            chunks_indexed=len(chunks),
            total_vectors=total_vectors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion error: {e}",
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run the multi-agent RAG pipeline and return the final answer",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Runs the MARA LangGraph for the given query and thread_id.
    Waits for full completion and returns a JSON response.
    """
    if not is_store_ready_for_thread(request.thread_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No documents found for thread '{request.thread_id}'. "
                "Please upload a PDF document first."
            ),
        )

    # Set context var so search_docs filters to this thread's documents only
    active_thread_id.set(request.thread_id)

    initial_state = make_initial_state(request.query, request.thread_id)
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_state = await _graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("Graph invocation failed for thread %s", request.thread_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent error: {e}",
        )

    # Extract the last substantive AI answer
    answer = ""
    from langchain_core.messages import AIMessage
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = msg.content
            break

    # Build step summaries from state history
    steps: list[StepSummary] = []
    try:
        for snapshot in _graph.get_state_history(config):
            node = snapshot.next[0] if snapshot.next else None
            if node and node in ("retriever", "reviewer"):
                steps.append(
                    StepSummary(
                        node=node,
                        output_summary=f"Node '{node}' executed (retry #{snapshot.values.get('retry_count', 0)})",
                    )
                )
    except Exception:
        pass  # step history is best-effort

    return QueryResponse(
        answer=answer or "No answer could be generated.",
        thread_id=request.thread_id,
        retry_count=final_state.get("retry_count", 0),
        review_passed=final_state.get("review_passed", False),
        steps=steps,
    )
