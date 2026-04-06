# MARA — Multi-Agent RAG Architect

A production-ready Retrieval-Augmented Generation (RAG) application built with **LangGraph**, **AWS Bedrock**, **ChromaDB**, **FastAPI**, and **Streamlit**. MARA uses a multi-agent pipeline to ensure answers are grounded in your documents and never hallucinated.

---

## Table of Contents

- [Architectural Overview](#architecture-overview)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [How Thread Isolation Works](#how-thread-isolation-works)
- [How the Review Loop Works](#how-the-review-loop-works)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                   │
│  • PDF upload sidebar    • Chat interface       │
│  • Thread ID management  • Agent thought log    │
└──────────────┬──────────────────────────────────┘
               │ HTTP (requests)
               ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend                    │
│  POST /upload   POST /query   GET /health       │
└──────┬───────────────────────────┬──────────────┘
       │                           │
       ▼                           ▼
┌─────────────┐          ┌─────────────────────────┐
│  ChromaDB   │          │    LangGraph Pipeline   │
│  (per-thread│◄─────────│  chat → retriever →     │
│  document   │ search   │  reviewer → END         │
│  store)     │          └──────────────┬──────────┘
└─────────────┘                         │
       ▲                                ▼
       │ embed               ┌─────────────────────┐
       │                     │  AsyncSqliteSaver   │
┌──────┴──────────┐          │  (conversation      │
│  AWS Bedrock    │          │   memory per thread)│
│  Titan Embed    │          └─────────────────────┘
│  Claude 3.5     │
│  Sonnet         │
└─────────────────┘
```

---

## Agent Pipeline

Every query passes through a 3-node LangGraph:

```
START
  │
  ▼
┌─────────────────────────────────────────────┐
│  CHAT NODE  (entry point)                   │
│  Reads full conversation history and decides│
│  • Greeting / follow-up → answer directly   │
│  • Needs documents → route to Retriever     │
└──────────────────┬──────────────────────────┘
        needs_retrieval?
       /              \
     YES               NO
      │                 │
      ▼                 ▼
┌──────────┐          END
│RETRIEVER │  (Agent A — ReAct)
│ Calls    │
│search_docs│
│ tool,    │
│ drafts   │
│ answer   │
└────┬─────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  REVIEWER  (Agent B — Reflection)            │
│  Scores answer on 4 criteria:                │
│  1. Groundedness  2. Completeness            │
│  3. Consistency   4. Precision               │
│  Outputs: PASS or FAIL + feedback            │
└───────────────────┬──────────────────────────┘
              PASS or retry≥3?
             /                \
           YES                 NO
            │                   │
           END          back to RETRIEVER
                        (with feedback, max 3 retries)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (Stateful Graph + AsyncSqliteSaver) |
| LLM | AWS Bedrock — Claude 3.5 Sonnet |
| Embeddings | AWS Bedrock — Amazon Titan Embed Text v1 |
| Vector Store | ChromaDB (persistent, per-thread filtered) |
| API | FastAPI (async) |
| Frontend | Streamlit |
| Memory | LangGraph AsyncSqliteSaver (SQLite, per-thread) |

---

## Project Structure

```
RAG_APP/
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # All settings via pydantic-settings + .env
│   ├── store.py             # ChromaDB singleton, per-thread doc storage
│   ├── engine.py            # LangGraph: state, nodes, graph definition
│   └── main.py              # FastAPI app, endpoints, lifespan
├── ui/
│   └── streamlit_app.py     # Streamlit chat interface
├── data/                    # Created at runtime
│   ├── chroma/              # ChromaDB persistent storage
│   └── checkpoints.db       # SQLite conversation memory
├── venv/                    # Python virtual environment
├── requirements.txt
├── .env.example             # Environment variable template
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access enabled for:
  - `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - `amazon.titan-embed-text-v1`

### 1. Create and activate the virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

---

## Configuration

All settings live in `.env`. Key options:

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | LLM model ID |
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v1` | Embedding model ID |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Where ChromaDB stores vectors |
| `CHROMA_COLLECTION_NAME` | `mara_docs` | ChromaDB collection name |
| `CHECKPOINT_DB_PATH` | `./data/checkpoints.db` | Conversation memory database |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_RETRIEVAL_DOCS` | `4` | Top-K docs returned per search |

> **Cross-region inference:** If you get a `ResourceNotFoundException`, prefix the model ID with `us.` — e.g. `us.anthropic.claude-3-5-sonnet-20241022-v2:0`

---

## Running the Application

Open **two terminals** with the virtual environment activated.

**Terminal 1 — FastAPI backend:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO: AsyncSqliteSaver initialised at ./data/checkpoints.db
INFO: MARA LangGraph compiled and ready
INFO: Application startup complete.
```

**Terminal 2 — Streamlit frontend:**

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Usage

1. **Set a Thread ID** in the sidebar (default: `mara-default`). Each thread has its own documents and conversation history.
2. **Upload a PDF** using the sidebar file uploader and click "Index Document".
3. **Ask questions** in the chat. The agent will retrieve relevant passages and review the answer before returning it.
4. **Switch threads** via the Thread ID field to start a fresh isolated session.

---

## API Reference

### `GET /health`

Returns backend status and whether the global document store has any vectors.

```json
{
  "status": "ok",
  "store_ready": true,
  "model": "anthropic.claude-3-5-sonnet-20241022-v2:0"
}
```

---

### `POST /upload`

Upload a PDF and index it into ChromaDB under the given thread.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF file to upload |
| `thread_id` | string | No (default: `"default"`) | Thread to associate documents with |

**Response:**

```json
{
  "message": "Successfully indexed 'document.pdf'",
  "chunks_indexed": 42,
  "total_vectors": 42
}
```

---

### `POST /query`

Run the multi-agent RAG pipeline and return the final answer.

**Request body:**

```json
{
  "query": "What are the coverage options?",
  "thread_id": "mara-default"
}
```

**Response:**

```json
{
  "answer": "According to the document...",
  "thread_id": "mara-default",
  "retry_count": 1,
  "review_passed": true,
  "steps": [
    { "node": "retriever", "output_summary": "Node 'retriever' executed (retry #0)" },
    { "node": "reviewer",  "output_summary": "Node 'reviewer' executed (retry #1)" }
  ]
}
```

**Error — no documents for thread:**

```json
{
  "detail": "No documents found for thread 'mara-default'. Please upload a PDF document first."
}
```

---

## How Thread Isolation Works

MARA maintains complete isolation between threads at two levels:

### Document isolation (ChromaDB)
Every uploaded chunk is tagged with `metadata["thread_id"]`. All similarity searches use a `filter={"thread_id": thread_id}` clause, so a query on Thread A never sees documents uploaded on Thread B.

### Conversation memory isolation (SQLite checkpointer)
LangGraph's `AsyncSqliteSaver` stores conversation state keyed by `thread_id`. Each thread has its own message history — follow-up questions work within a thread but don't bleed across threads.

```
Thread A                          Thread B
├── docs: contract.pdf            ├── docs: manual.pdf
└── memory: "What is clause 3?"  └── memory: "How do I reset?"
         "It states that..."               "Press the button..."
```

---

## How the Review Loop Works

The Reviewer evaluates every retrieval-based answer against four criteria:

| Criterion | What it checks |
|---|---|
| **Groundedness** | Every factual claim appears in the retrieved context |
| **Completeness** | All parts of the question are addressed |
| **Consistency** | No internal contradictions or contradictions with context |
| **Precision** | No evaluative claims (e.g. "best", "recommended") absent from context |

If the answer **fails**, the reviewer appends specific feedback to the message history and the Retriever runs again — up to **3 retries**. After 3 retries the best available answer is returned regardless.

The Reviewer always evaluates the **current question** (last user message) using the full conversation history for context, so follow-up questions are reviewed correctly.
