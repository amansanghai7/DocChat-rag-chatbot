import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import db_service
from langgraph_rag_backend import (
    chatbot,
    create_thread_metadata,
    delete_thread_completely,
    generate_chat_title,
    get_all_threads_with_metadata,
    get_pinecone_stats,
    get_thread_title,
    ingest_pdf,
    thread_document_metadata,
    update_thread_title,
)


# =========================== Message Rendering ===========================
def render_message(message, role=None):
    """
    Unified message rendering function for both streaming and restored messages.
    Handles markdown, code blocks, and multiline text properly.
    """
    if isinstance(message, dict):
        content = message.get("content", "")
        role = message.get("role", role)
    elif isinstance(message, (AIMessage, HumanMessage, ToolMessage)):
        content = message.content
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "tool"
    else:
        content = str(message)
    
    # Handle empty content
    if not content:
        content = "_[Empty message]_"
    
    # Render with markdown support
    st.markdown(content)


def format_tool_message(tool_msg: ToolMessage) -> str:
    """Format tool messages nicely for display."""
    tool_name = getattr(tool_msg, "name", "unknown_tool")
    content = tool_msg.content
    
    # Try to format as code block if it looks like JSON/dict
    if isinstance(content, (dict, list)):
        import json
        content = json.dumps(content, indent=2)
        return f"**🔧 Tool: `{tool_name}`**\n```json\n{content}\n```"
    
    return f"**🔧 Tool: `{tool_name}`**\n```\n{content}\n```"


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    """Create a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    create_thread_metadata(thread_id, "New Chat")
    st.session_state["message_history"] = []
    st.session_state["title_generated"] = False


def load_conversation(thread_id):
    """Load conversation history from LangGraph checkpoint."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    
    # Convert to display format, filtering out tool messages
    display_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            # Only add if it has content (not just tool calls)
            if msg.content:
                display_messages.append({"role": "assistant", "content": msg.content})
    
    return display_messages


def count_user_messages():
    """Count how many user messages exist in current conversation."""
    return sum(1 for msg in st.session_state["message_history"] if msg["role"] == "user")


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    create_thread_metadata(thread_id, "New Chat")

if "title_generated" not in st.session_state:
    st.session_state["title_generated"] = False

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "delete_confirm" not in st.session_state:
    st.session_state["delete_confirm"] = None  # Stores thread_id to delete

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
current_title = get_thread_title(thread_key)

# ============================ Sidebar ============================
st.sidebar.title("🤖 LangGraph PDF Chatbot")

# Display current chat title
st.sidebar.markdown(f"**Current Chat:** {current_title}")
st.sidebar.caption(f"Thread ID: `{thread_key[:8]}...`")

# Pinecone status (optional, can be collapsed)
with st.sidebar.expander("📊 Pinecone Status", expanded=False):
    try:
        stats = get_pinecone_stats()
        if "error" not in stats:
            st.metric("Total Vectors", stats.get("total_vector_count", 0))
            st.metric("Active Threads", len(stats.get("namespaces", {})))
            st.caption(f"Index Fullness: {stats.get('index_fullness', 0):.2%}")
        else:
            st.error(f"Error: {stats['error']}")
    except Exception as e:
        st.warning(f"Could not fetch Pinecone stats: {e}")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# PDF upload section
st.sidebar.divider()
st.sidebar.subheader("📄 Document Upload")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"✅ **{latest_doc.get('filename')}**\n\n"
        f"📊 {latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed yet for this chat.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], key=f"uploader_{thread_key}")
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed.")
    else:
        with st.sidebar.status("🔄 Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed successfully", state="complete", expanded=False)

# Past conversations section
st.sidebar.divider()
st.sidebar.subheader("💬 Conversations")

threads_with_metadata = get_all_threads_with_metadata()

if not threads_with_metadata:
    st.sidebar.write("No past conversations yet.")
else:
    # Show confirmation dialog if delete is pending
    if st.session_state["delete_confirm"] is not None:
        thread_to_delete = st.session_state["delete_confirm"]
        thread_title = get_thread_title(thread_to_delete)
        
        st.sidebar.warning(f"⚠️ Delete '{thread_title}'?")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("✅ Yes", key="confirm_delete", use_container_width=True):
                # Perform deletion
                result = delete_thread_completely(thread_to_delete)
                
                if result["success"]:
                    st.sidebar.success("🗑️ Thread deleted!")
                    
                    # Smart thread switching logic (ChatGPT-like)
                    if thread_to_delete == thread_key:
                        # Deleting current thread - need to switch
                        
                        # Get remaining threads (excluding the one being deleted)
                        remaining_threads = [t for t in threads_with_metadata if t["thread_id"] != thread_to_delete]
                        
                        if remaining_threads:
                            # Switch to the most recent remaining thread
                            next_thread = remaining_threads[0]  # Already sorted by updated_at DESC
                            st.session_state["thread_id"] = next_thread["thread_id"]
                            st.session_state["message_history"] = load_conversation(next_thread["thread_id"])
                            st.session_state["title_generated"] = True
                            st.session_state["ingested_docs"].setdefault(str(next_thread["thread_id"]), {})
                        else:
                            # No remaining threads - create new chat
                            reset_chat()
                    # else: deleting non-current thread, stay on current thread
                    
                    # Clear confirmation state
                    st.session_state["delete_confirm"] = None
                    st.rerun()
                else:
                    # Show user-friendly error message
                    error_summary = "Deletion completed with some issues"
                    if result.get("errors"):
                        # Filter out technical details for user
                        user_friendly_errors = []
                        for error in result["errors"]:
                            if "Pinecone" in error and ("not found" in error.lower() or "404" in error):
                                continue  # Skip "not found" errors - they're OK
                            user_friendly_errors.append(error)
                        
                        if user_friendly_errors:
                            st.sidebar.error(f"❌ {error_summary}")
                            for err in user_friendly_errors[:2]:  # Show max 2 errors
                                st.sidebar.caption(f"• {err[:100]}")
                        else:
                            # All errors were "not found" - actually successful
                            st.sidebar.success("🗑️ Thread deleted!")
                            
                            # Apply same smart switching logic
                            if thread_to_delete == thread_key:
                                remaining_threads = [t for t in threads_with_metadata if t["thread_id"] != thread_to_delete]
                                if remaining_threads:
                                    next_thread = remaining_threads[0]
                                    st.session_state["thread_id"] = next_thread["thread_id"]
                                    st.session_state["message_history"] = load_conversation(next_thread["thread_id"])
                                    st.session_state["title_generated"] = True
                                    st.session_state["ingested_docs"].setdefault(str(next_thread["thread_id"]), {})
                                else:
                                    reset_chat()
                    
                    st.session_state["delete_confirm"] = None
                    st.rerun()
        
        with col2:
            if st.button("❌ No", key="cancel_delete", use_container_width=True):
                st.session_state["delete_confirm"] = None
                st.rerun()
    
    # Display threads with delete buttons
    for thread_info in threads_with_metadata:
        thread_id = thread_info["thread_id"]
        chat_title = thread_info["chat_title"]
        
        # Create columns for thread button and delete button
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            # Highlight current thread
            if thread_id == thread_key:
                button_label = f"🟢 {chat_title}"
            else:
                button_label = chat_title
            
            if st.button(
                button_label,
                key=f"thread-btn-{thread_id}",
                use_container_width=True
            ):
                # Switch to this thread
                st.session_state["thread_id"] = thread_id
                st.session_state["message_history"] = load_conversation(thread_id)
                st.session_state["title_generated"] = True  # Assume old threads have titles
                st.session_state["ingested_docs"].setdefault(str(thread_id), {})
                st.rerun()
        
        with col2:
            # Delete button
            if st.button("🗑️", key=f"delete-btn-{thread_id}", help="Delete this conversation"):
                st.session_state["delete_confirm"] = thread_id
                st.rerun()

# ============================ Main Layout ========================
st.title("💬 Multi Utility Chatbot")
st.caption("Ask questions about your documents or use built-in tools")

# Display chat history with proper rendering
for message in st.session_state["message_history"]:
    role = message["role"]
    with st.chat_message(role):
        render_message(message, role=role)

# Chat input
user_input = st.chat_input("Ask about your document or use tools...")

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        render_message({"role": "user", "content": user_input})

    # Persist user message to Supabase
    db_service.save_message(thread_key, "user", user_input)

    # Check if we need to generate title (before streaming)
    should_generate_title = not st.session_state["title_generated"] and count_user_messages() == 1

    # Prepare config for LangGraph
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            """Stream only AI message content, handle tool messages separately."""
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        # Stream and render with markdown
        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution complete", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Persist AI response to Supabase
    db_service.save_message(thread_key, "assistant", ai_message)

    # Generate title after first complete conversation turn
    if should_generate_title:
        new_title = generate_chat_title(user_input)
        update_thread_title(thread_key, new_title)
        st.session_state["title_generated"] = True
        # Rerun to update sidebar with new title
        st.rerun()

    # Show document metadata if available
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Document: {doc_meta.get('filename')} "
            f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
        )