from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any

from langchain_aws import ChatBedrock
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app import store as vector_store
from app.config import settings
from app.store import active_thread_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation + tool call history
    context: str                              # retrieved document snippets
    review_passed: bool                       # Reviewer verdict
    retry_count: int                          # loop-guard counter
    needs_retrieval: bool                     # set by chat_node to route the question
    thread_id: str                            # owning thread — used for doc filtering


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(temperature: float = 0.0) -> ChatBedrock:
    return ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        model_kwargs={"temperature": temperature, "max_tokens": 2048},
    )


# ---------------------------------------------------------------------------
# Tool: request_document_search  (used by chat_node to signal retrieval needed)
# ---------------------------------------------------------------------------

@tool
def request_document_search(reason: str) -> str:
    """
    Call this tool when the user's question requires searching the document
    knowledge base for new information.
    Do NOT call this for greetings, follow-up clarifications, or questions
    that can be answered from the conversation history.
    Args:
        reason: One-sentence explanation of why a document search is needed.
    """
    return reason  # return value is unused; tool call presence is what matters


# ---------------------------------------------------------------------------
# Tool: search_docs  (used by retriever_node)
# ---------------------------------------------------------------------------

@tool
async def search_docs(query: str) -> str:
    """
    Search the document knowledge base for relevant information.
    Args:
        query: A search query string describing what information is needed.
    Returns:
        Concatenated document snippets with source metadata.
    """
    tid = active_thread_id.get()
    docs = await vector_store.similarity_search(query, k=settings.max_retrieval_docs, thread_id=tid)
    if not docs:
        return "No relevant documents found for this query."
    snippets = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        snippets.append(f"[Source {i}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(snippets)


# ---------------------------------------------------------------------------
# Chat Node — entry point, decides direct answer vs. retrieval
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """You are a helpful conversational assistant for a document Q&A system.

For every user message decide ONE of two paths:

PATH 1 — Answer directly (no tool call needed):
  • Greetings, thanks, small talk
  • Follow-up questions like "can you explain that differently?" or "what did you mean by X?"
  • Questions fully answerable from the conversation history above
  → Just write your answer as normal text.

PATH 2 — Search the knowledge base (call request_document_search):
  • Questions that ask about document content you haven't retrieved yet
  • "What does the document say about...", "How does X work according to...", etc.
  → Call the request_document_search tool with a one-sentence reason.
  → Do NOT attempt to answer yourself; the retriever agent will handle it.

Be decisive. When in doubt about whether you already have the information, prefer PATH 2.
"""

async def chat_node(state: RAGState) -> dict[str, Any]:
    """
    Entry-point node. Reads conversation history and the latest question,
    then decides:
      - answer directly (needs_retrieval=False, review_passed=True → END), or
      - hand off to retriever (needs_retrieval=True → retriever node).
    """
    llm = make_llm(temperature=0.1)
    llm_with_tool = llm.bind_tools([request_document_search])

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + list(state["messages"])
    response: AIMessage = await llm_with_tool.ainvoke(messages)

    if response.tool_calls:
        # LLM decided retrieval is needed — don't pollute history with the
        # tool-call message; just flip the flag and let the retriever run.
        logger.info("Chat node → retrieval required")
        return {"needs_retrieval": True}
    else:
        # LLM answered directly from conversation context
        logger.info("Chat node → direct answer (no retrieval)")
        return {
            "messages": [response],
            "needs_retrieval": False,
            "review_passed": True,   # direct answers skip the review loop
        }


def route_from_chat(state: RAGState) -> str:
    """Routes after chat_node: go to retriever or end directly."""
    return "retriever" if state.get("needs_retrieval") else "end"


# ---------------------------------------------------------------------------
# Retriever Node — Agent A (ReAct style)
# ---------------------------------------------------------------------------

RETRIEVER_SYSTEM_PROMPT = """You are a precise document retrieval agent for a RAG system.

Your responsibilities:
1. Analyse the user's question to identify key concepts.
2. Use the search_docs tool to retrieve relevant document passages.
3. Draft a comprehensive answer STRICTLY based on the retrieved passages.
4. If retrieved passages are insufficient, explicitly state what information is missing.

IMPORTANT RULES:
- Do NOT use any knowledge outside the retrieved documents.
- Always call search_docs at least once before drafting your answer.
- Write your final response as a clean, direct answer only.
  Do NOT include headers, labels, or the raw retrieved text in your response.
"""

async def retriever_node(state: RAGState) -> dict[str, Any]:
    """
    Agent A: Retrieves relevant documents and drafts an initial answer.
    Implements a tool-call loop (max 3 tool calls to prevent runaway).
    Writes back: messages (AI + ToolMessages), context (raw snippets).
    """
    llm = make_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools([search_docs])

    base_messages = [SystemMessage(content=RETRIEVER_SYSTEM_PROMPT)] + list(state["messages"])
    new_messages: list = []
    retrieved_context = ""
    tool_call_count = 0
    MAX_TOOL_CALLS = 3

    while tool_call_count < MAX_TOOL_CALLS:
        ai_msg: AIMessage = await llm_with_tools.ainvoke(base_messages + new_messages)
        new_messages.append(ai_msg)

        if not ai_msg.tool_calls:
            break

        for tc in ai_msg.tool_calls:
            if tc["name"] == "search_docs":
                result: str = await search_docs.ainvoke(tc["args"])
                retrieved_context += result + "\n\n"
                new_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
        tool_call_count += 1

    logger.info("Retriever finished — %d tool calls, context length %d", tool_call_count, len(retrieved_context))
    return {
        "messages": new_messages,
        "context": retrieved_context.strip(),
    }


# ---------------------------------------------------------------------------
# Reviewer Node — Agent B (Reflection)
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """You are a strict factual accuracy reviewer for a RAG system.

You will receive:
1. The recent conversation history (for follow-up context)
2. The current user question being answered
3. The retrieved document context
4. The proposed answer from the Retrieval Agent

Evaluate the proposed answer against these four criteria:
1. GROUNDEDNESS   — Every factual claim must be present (verbatim or paraphrased) in the context.
                    FAIL if any claim has no context support.
2. COMPLETENESS   — All parts of the CURRENT question must be addressed.
                    FAIL if any sub-question is ignored.
3. CONSISTENCY    — No internal contradictions and no contradiction with context.
                    FAIL if contradictions exist.
4. PRECISION      — No evaluative judgments absent from the context (e.g. "best", "recommended").
                    FAIL if the answer introduces qualitative claims the context doesn't make.

You MUST NOT use any external knowledge. Judge only against the provided context.
Use the conversation history only to understand what the current question refers to.

Respond ONLY with a single JSON object — no prose, no markdown fences:
{"verdict": "PASS", "reason": "...", "feedback": ""}
or
{"verdict": "FAIL", "reason": "...", "feedback": "Specific instructions for the retriever to improve the answer"}
"""

async def reviewer_node(state: RAGState) -> dict[str, Any]:
    """
    Agent B: Evaluates the Retriever's answer for hallucination and completeness.
    Writes back: review_passed (bool), retry_count (+1),
                 messages (appends HumanMessage feedback if FAIL).
    """
    llm = make_llm(temperature=0.0)

    # Find the LAST user question that isn't reviewer feedback — this is the
    # question the retriever just answered, not the first message in history.
    current_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("[REVIEWER FEEDBACK"):
            current_question = msg.content
            break

    # Build a concise conversation history (last 6 messages, excluding tool messages
    # and reviewer feedback) so the reviewer understands follow-up context.
    history_lines: list[str] = []
    recent = [
        m for m in state["messages"]
        if not (isinstance(m, HumanMessage) and m.content.startswith("[REVIEWER FEEDBACK"))
        and not isinstance(m, ToolMessage)
    ]
    for msg in recent[-6:]:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            history_lines.append(f"Assistant: {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")
    conversation_history = "\n".join(history_lines) if history_lines else "No prior history."

    # Extract the last substantive AI answer (skip tool-call AIMessages)
    proposed_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            proposed_answer = msg.content
            break

    review_prompt = (
        f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
        f"CURRENT QUESTION:\n{current_question}\n\n"
        f"RETRIEVED CONTEXT:\n{state.get('context', 'No context available')}\n\n"
        f"PROPOSED ANSWER:\n{proposed_answer}\n\n"
        "Apply your four criteria to the CURRENT QUESTION and respond with the JSON verdict."
    )

    response: AIMessage = await llm.ainvoke(
        [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=review_prompt),
        ]
    )

    # Robust JSON parsing — handles markdown code fences
    raw = response.content
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    verdict_data: dict = {"verdict": "PASS", "reason": "parse-error fallback", "feedback": ""}
    if json_match:
        try:
            verdict_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Reviewer JSON parse failed; defaulting to PASS. Raw: %s", raw[:200])

    passed = verdict_data.get("verdict", "PASS").upper() == "PASS"
    new_messages: list = []

    if not passed:
        feedback = verdict_data.get("feedback") or "Please re-examine the context and improve the answer."
        new_messages.append(
            HumanMessage(
                content=(
                    f"[REVIEWER FEEDBACK — Attempt {state['retry_count'] + 1}]: "
                    f"{feedback}"
                )
            )
        )
        logger.info("Reviewer: FAIL — %s", verdict_data.get("reason", ""))
    else:
        logger.info("Reviewer: PASS — %s", verdict_data.get("reason", ""))

    return {
        "review_passed": passed,
        "retry_count": state["retry_count"] + 1,
        "messages": new_messages,
    }


# ---------------------------------------------------------------------------
# Router: after reviewer
# ---------------------------------------------------------------------------

def route_after_review(state: RAGState) -> str:
    """End if review passed OR max retries reached (retry_count already incremented)."""
    if state["review_passed"] or state["retry_count"] >= 3:
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer) -> Any:
    """
    Constructs and compiles the MARA LangGraph.

    Topology:
        START → chat_node ──[needs_retrieval=False]──────────────► END
                    │
                    └──[needs_retrieval=True]──► retriever → reviewer ─[passed/retry>=3]─► END
                                                     ▲            │
                                                     └──[retry]───┘
    """
    graph = StateGraph(RAGState)

    graph.add_node("chat", chat_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("chat")
    graph.add_conditional_edges(
        "chat",
        route_from_chat,
        {"retriever": "retriever", "end": END},
    )
    graph.add_edge("retriever", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"end": END, "retry": "retriever"},
    )

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# State initialiser
# ---------------------------------------------------------------------------

def make_initial_state(query: str, thread_id: str = "default") -> RAGState:
    """Creates a fresh RAGState for a new query invocation."""
    return RAGState(
        messages=[HumanMessage(content=query)],
        context="",
        review_passed=False,
        retry_count=0,
        needs_retrieval=False,
        thread_id=thread_id,
    )
