# Initial requirements
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MARA — Multi-Agent RAG",
    page_icon="🏛",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "messages": [],       # [{"role": "user"|"assistant", "content": str, "steps": list}]
        "thread_id": "mara-default",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("MARA Configuration")
        st.divider()

        # Thread ID
        st.subheader("Session")
        new_thread = st.text_input(
            "Thread ID",
            value=st.session_state.thread_id,
            help="Change to start a fresh conversation with no memory of prior exchanges.",
        )
        if new_thread != st.session_state.thread_id:
            st.session_state.thread_id = new_thread
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Document upload
        st.subheader("Knowledge Base")
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF to add to the ChromaDB knowledge base.",
        )

        if uploaded_file and st.button("Index Document", type="primary"):
            with st.spinner(f"Indexing {uploaded_file.name} ..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        data={"thread_id": st.session_state.thread_id},
                        timeout=120,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"Indexed **{data['chunks_indexed']}** chunks "
                            f"({data['total_vectors']} total vectors)"
                        )
                    else:
                        detail = response.json().get("detail", "Unknown error")
                        st.error(f"Upload failed: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to MARA backend. Is it running on port 8000?")
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. The PDF may be too large or Bedrock is slow.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        st.divider()

        # Health check
        if st.button("Check Backend Status"):
            try:
                r = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if r.status_code == 200:
                    h = r.json()
                    store_status = "Ready" if h.get("store_ready") else "Empty (upload a PDF)"
                    st.markdown(f"**Status:** {h.get('status', 'unknown')}")
                    st.markdown(f"**Knowledge base:** {store_status}")
                    st.markdown(f"**Model:** `{h.get('model', 'unknown')}`")
                else:
                    st.warning(f"Backend returned HTTP {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Backend unreachable on port 8000.")
            except Exception as e:
                st.error(f"Health check failed: {e}")


# ---------------------------------------------------------------------------
# Agent steps expander
# ---------------------------------------------------------------------------
def render_agent_steps(steps: list[dict]):
    if not steps:
        return
    with st.expander("Agent Thought Process", expanded=False):
        for step in steps:
            node = step.get("node", "unknown")
            summary = step.get("output_summary", "")
            icon = "🔍" if node == "retriever" else "✅" if node == "reviewer" else "⚙"
            st.markdown(f"{icon} **{node.capitalize()}** — {summary}")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def call_query_api(query: str, thread_id: str) -> dict:
    """Calls POST /query and returns the parsed JSON response dict."""
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"query": query, "thread_id": thread_id},
        timeout=(10, 300),   # 10s connect, 300s read
    )
    response.raise_for_status()
    return response.json()


def render_chat():
    st.title("MARA — Multi-Agent RAG Architect")
    st.caption(
        "Powered by AWS Bedrock (Claude 3.5 Sonnet) · LangGraph · ChromaDB"
    )

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                render_agent_steps(msg["steps"])

    # Accept new user input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input, "steps": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agents and display response
        with st.chat_message("assistant"):
            with st.spinner("Running MARA agents..."):
                try:
                    data = call_query_api(user_input, st.session_state.thread_id)

                    answer = data.get("answer", "No answer generated.")
                    retry_count = data.get("retry_count", 0)
                    review_passed = data.get("review_passed", False)
                    steps = data.get("steps", [])

                    st.markdown(answer)

                    # Retry / review badge
                    if retry_count > 0:
                        verdict_icon = "✅" if review_passed else "⚠️"
                        st.caption(
                            f"{verdict_icon} Review: "
                            f"{'Passed' if review_passed else 'Max retries reached'} "
                            f"after {retry_count} attempt(s)"
                        )

                    render_agent_steps(steps)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "steps": steps,
                        }
                    )

                except requests.exceptions.ConnectionError:
                    msg = "Cannot connect to MARA backend. Make sure `uvicorn app.main:app` is running on port 8000."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {msg}", "steps": []})

                except requests.exceptions.Timeout:
                    msg = "Query timed out (300s). Try a shorter or more focused question."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {msg}", "steps": []})

                except requests.exceptions.HTTPError as e:
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    st.error(f"Backend error: {detail}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {detail}", "steps": []})

                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "steps": []})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
