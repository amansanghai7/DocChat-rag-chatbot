import os
import uuid

import requests
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

# =========================== Authentication ==============================
# Supabase project URL and key are read from environment variables.
# The service-role key is safe here because Streamlit runs server-side —
# it is never sent to the browser.
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _login_with_supabase(email: str, password: str) -> dict | None:
    """
    POST to Supabase Auth REST API with email + password.
    Returns the full auth response dict on success, or None on failure.

    Supabase returns: { access_token, user: { id, email, ... }, ... }
    The access_token is an ES256-signed JWT — same token FastAPI verifies.
    """
    try:
        resp = requests.post(
            f"{_SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={"email": email, "password": password},
            headers={"apikey": _SUPABASE_KEY, "Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None


def _signup_with_supabase(email: str, password: str) -> dict:
    """
    POST to Supabase Auth signup endpoint to create a new user account.

    Signup differs from login:
      - Login:  user already exists → Supabase verifies password → returns JWT
      - Signup: user does not exist → Supabase creates account → returns user object

    Returns a result dict with keys:
      { "status": "logged_in" | "confirm_email" | "error", "data": ..., "message": ... }

    "logged_in"     — account created + session returned (email confirmation OFF in Supabase)
    "confirm_email" — account created but awaiting email confirmation (confirmation ON)
    "error"         — duplicate email, weak password, or network failure
    """
    try:
        resp = requests.post(
            f"{_SUPABASE_URL}/auth/v1/signup",
            json={"email": email, "password": password},
            headers={"apikey": _SUPABASE_KEY, "Content-Type": "application/json"},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code == 200:
            # Supabase returns access_token only when email confirmation is disabled.
            if data.get("access_token"):
                return {"status": "logged_in", "data": data}
            # User created but must confirm email before logging in.
            return {"status": "confirm_email", "data": data}

        # Supabase returns error details in the "msg" field.
        raw_msg = data.get("msg", data.get("message", "Signup failed."))

        # Translate Supabase's internal messages into user-friendly ones.
        if "already registered" in raw_msg.lower():
            msg = "An account with this email already exists. Please sign in instead."
        elif "password" in raw_msg.lower():
            msg = "Password must be at least 6 characters long."
        elif "invalid" in raw_msg.lower() and "email" in raw_msg.lower():
            msg = "Please enter a valid email address."
        else:
            msg = raw_msg

        return {"status": "error", "message": msg}

    except requests.RequestException:
        return {"status": "error", "message": "Network error. Please check your connection."}


def _show_auth_page() -> None:
    """
    Combined authentication screen that renders either the login form or the
    signup form depending on st.session_state["auth_mode"].

    Mode switching works by changing auth_mode and calling st.rerun().
    Streamlit reruns the script top-to-bottom on each interaction — the new
    mode value in session_state is read immediately on the next run.

    st.session_state persists for the browser tab's lifetime.
    Closing the tab clears it — the user must log in again, but all their
    conversation data remains permanently stored in Supabase.
    """
    st.title("🤖 DocChat — RAG Chatbot")
    st.divider()

    # Default to login screen if mode has not been set yet.
    mode = st.session_state.get("auth_mode", "login")

    # ── Login form ────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown("### Sign in")

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Signing in…"):
                    result = _login_with_supabase(email.strip(), password)

                if result:
                    st.session_state["token"] = result["access_token"]
                    st.session_state["user_id"] = result["user"]["id"]
                    st.session_state["user_email"] = result["user"]["email"]
                    st.rerun()  # gate passes on next run
                else:
                    st.error("Invalid email or password. Please try again.")

        st.markdown("---")
        st.markdown("Don't have an account?")
        if st.button("Create account", use_container_width=True):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    # ── Signup form ───────────────────────────────────────────────────────────
    else:
        st.markdown("### Create account")

        with st.form("signup_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Min. 6 characters")
            confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
            submitted = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            # Client-side validation before hitting Supabase.
            if not email or not password or not confirm:
                st.error("All fields are required.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating your account…"):
                    result = _signup_with_supabase(email.strip(), password)

                if result["status"] == "logged_in":
                    # Email confirmation is disabled in Supabase — account created
                    # and session returned in one step. Auto-login the user.
                    data = result["data"]
                    st.session_state["token"] = data["access_token"]
                    st.session_state["user_id"] = data["user"]["id"]
                    st.session_state["user_email"] = data["user"]["email"]
                    st.success("Account created! Welcome.")
                    st.rerun()

                elif result["status"] == "confirm_email":
                    # Supabase sent a confirmation email. User must verify before
                    # they can log in. Switch to login screen with a hint.
                    st.success("Account created! Check your email to confirm, then sign in.")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

                else:
                    st.error(result["message"])

        st.markdown("---")
        st.markdown("Already have an account?")
        if st.button("Back to sign in", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()


# ── Authentication gate ───────────────────────────────────────────────────────
# If no token in session_state, show the auth page and stop script execution.
# Nothing below this block runs for unauthenticated users.
if "token" not in st.session_state:
    _show_auth_page()
    st.stop()

# From here: user is authenticated.
# These values are safe to read on every rerun.
_user_id: str = st.session_state["user_id"]
_user_email: str = st.session_state["user_email"]

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
    """Create a new chat thread owned by the current user."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    create_thread_metadata(thread_id, "New Chat", user_id=st.session_state.get("user_id"))
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
    create_thread_metadata(thread_id, "New Chat", user_id=st.session_state.get("user_id"))

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

# ── User identity + logout ────────────────────────────────────────────────────
st.sidebar.caption(f"Signed in as **{_user_email}**")
if st.sidebar.button("Sign out", use_container_width=True):
    # Clear all session state — the gate will show the login page on next rerun.
    st.session_state.clear()
    st.rerun()
st.sidebar.divider()

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

threads_with_metadata = get_all_threads_with_metadata(user_id=_user_id)

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
        # Rule: zero st.* calls inside the generator passed to write_stream.
        # Any st.* call inside the generator triggers an implicit Streamlit
        # rerun, which clears user_input so the title-generation block after
        # this with-block is never reached.
        # Rule: zero st.* calls after write_stream inside this with-block.
        # Calling st.empty().empty() or st.status() after write_stream also
        # triggers a re-render that can fire an implicit rerun for the same
        # reason (observed to break title generation for non-tool responses).
        # Solution: collect tool names during streaming (no widget side-effects),
        # then render tool status in a SEPARATE with-block after this one.
        tool_calls_made: list[str] = []

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_calls_made.append(getattr(message_chunk, "name", "tool"))
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        raw_output = st.write_stream(ai_only_stream())

    # Tool status is rendered OUTSIDE the assistant chat_message block so that
    # no widget calls happen inside the streaming context.
    if tool_calls_made:
        with st.chat_message("assistant"):
            with st.status("✅ Tool execution complete", state="complete"):
                for name in tool_calls_made:
                    st.caption(f"🔧 {name}")

    # Normalize write_stream output to a plain string (guard for edge cases)
    ai_message = raw_output if isinstance(raw_output, str) else ""

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
        st.rerun()

    # Show document metadata if available
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Document: {doc_meta.get('filename')} "
            f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
        )