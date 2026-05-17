"""
Database service layer — all Supabase CRUD operations live here.

Design principles:
  - Every function is a thin wrapper around one Supabase table operation.
  - No business logic — that stays in langgraph_rag_backend.py.
  - Every function returns a plain Python value (dict / list / bool / str).
  - Errors are caught, logged, and never bubble up to crash the app.
  - Future FastAPI routes or FastMCP tools can import these functions directly.

Tables (already created in Supabase):
  threads   — one row per conversation (id, title, created_at, updated_at)
  messages  — one row per user/AI turn (id, thread_id fk, role, content, created_at)
  documents — one row per uploaded PDF (id, thread_id fk, filename, pinecone_namespace, upload_time)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string accepted by Supabase."""
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# THREAD operations
# ─────────────────────────────────────────────────────────────────────────────

def create_thread(thread_id: str, title: str = "New Chat", user_id: Optional[str] = None) -> dict:
    """
    Insert a new thread row.

    Uses upsert with ignore_duplicates=True so calling this twice for the
    same thread_id is always safe (acts like INSERT OR IGNORE in SQLite).
    Returns the created row dict, or {} on failure.

    user_id: Supabase auth user UUID. When provided the row is tied to that
             user so all subsequent queries can be scoped by owner.
    """
    try:
        data: Dict[str, Any] = {"id": thread_id, "title": title}
        if user_id:
            data["user_id"] = user_id
        response = (
            supabase.table("threads")
            .upsert(data, ignore_duplicates=True)
            .execute()
        )
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error("create_thread(%s) failed: %s", thread_id, e)
        return {}


def get_thread_title(thread_id: str) -> str:
    """
    Fetch the display title for a thread.
    Returns 'New Chat' if the thread doesn't exist or on any error.
    """
    try:
        response = (
            supabase.table("threads")
            .select("title")
            .eq("id", thread_id)
            .maybe_single()
            .execute()
        )
        if response and response.data:
            return response.data.get("title") or "New Chat"
        return "New Chat"
    except Exception as e:
        logger.warning("get_thread_title(%s) failed: %s", thread_id, e)
        return "New Chat"


def update_thread_title(thread_id: str, title: str) -> bool:
    """
    Update the title and bump updated_at for a thread.
    Returns True on success, False on failure.
    """
    try:
        supabase.table("threads").update(
            {"title": title, "updated_at": _utcnow()}
        ).eq("id", thread_id).execute()
        return True
    except Exception as e:
        logger.error("update_thread_title(%s) failed: %s", thread_id, e)
        return False


def touch_thread(thread_id: str) -> None:
    """
    Bump updated_at for a thread without changing its title.
    Called after every message so the sidebar stays sorted by last activity.
    """
    try:
        supabase.table("threads").update(
            {"updated_at": _utcnow()}
        ).eq("id", thread_id).execute()
    except Exception as e:
        logger.warning("touch_thread(%s) failed: %s", thread_id, e)


def get_all_threads(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return threads ordered by updated_at DESC (most recent first).

    When user_id is provided only that user's threads are returned.
    When user_id is None all threads are returned (legacy / Streamlit path).

    Returns a list of dicts with keys:
        thread_id, chat_title, created_at, updated_at
    (Legacy-compatible key names so the frontend needs no changes.)
    """
    try:
        query = (
            supabase.table("threads")
            .select("id, title, created_at, updated_at")
            .order("updated_at", desc=True)
        )
        if user_id:
            query = query.eq("user_id", user_id)
        response = query.execute()
        threads = []
        for row in (response.data or []):
            threads.append(
                {
                    "thread_id": row["id"],
                    "chat_title": row.get("title") or "New Chat",
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            )
        return threads
    except Exception as e:
        logger.error("get_all_threads() failed: %s", e)
        return []


def delete_thread(thread_id: str) -> bool:
    """
    Hard-delete a thread row.
    The ON DELETE CASCADE in Supabase automatically deletes all related
    messages and documents rows — no manual cleanup needed.
    Returns True on success, False on failure.
    """
    try:
        supabase.table("threads").delete().eq("id", thread_id).execute()
        return True
    except Exception as e:
        logger.error("delete_thread(%s) failed: %s", thread_id, e)
        return False


def get_thread(thread_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Return full thread data for a single thread_id, or None if not found.
    Returns dict with keys: thread_id, title, created_at, updated_at.

    When user_id is provided the query also filters by owner, so a thread
    belonging to a different user returns None (same as not found).
    This prevents cross-user data leakage without a separate ownership check.
    """
    try:
        query = (
            supabase.table("threads")
            .select("id, title, created_at, updated_at")
            .eq("id", thread_id)
        )
        if user_id:
            query = query.eq("user_id", user_id)
        response = query.maybe_single().execute()
        if not response.data:
            return None
        row = response.data
        return {
            "thread_id": row["id"],
            "title": row.get("title") or "New Chat",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }
    except Exception as e:
        logger.error("get_thread(%s) failed: %s", thread_id, e)
        return None


def thread_exists(thread_id: str, user_id: Optional[str] = None) -> bool:
    """
    Return True if a thread row exists for this ID.
    When user_id is provided also verifies the thread belongs to that user.
    """
    try:
        query = supabase.table("threads").select("id").eq("id", thread_id)
        if user_id:
            query = query.eq("user_id", user_id)
        response = query.maybe_single().execute()
        return response.data is not None
    except Exception as e:
        logger.warning("thread_exists(%s) failed: %s", thread_id, e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE operations
# ─────────────────────────────────────────────────────────────────────────────

def save_message(thread_id: str, role: str, content: str) -> dict:
    """
    Persist a single message turn to the messages table and bump the
    thread's updated_at so the sidebar stays sorted by last activity.

    Args:
        thread_id: UUID of the owning thread.
        role:      'user' or 'assistant'.
        content:   Plain-text message body.

    Returns the inserted row dict, or {} on failure.
    """
    try:
        response = (
            supabase.table("messages")
            .insert({"thread_id": thread_id, "role": role, "content": content})
            .execute()
        )
        # Keep threads sorted by last activity
        touch_thread(thread_id)
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error("save_message(thread=%s, role=%s) failed: %s", thread_id, role, e)
        return {}


def get_messages(thread_id: str) -> List[Dict[str, Any]]:
    """
    Return all messages for a thread in chronological order.

    Returns list of dicts with keys: role, content, created_at.
    Useful for building read-only chat history views or future API endpoints.
    Note: LangGraph still owns the authoritative conversation state via its
    own checkpoint tables — this table is a readable mirror.
    """
    try:
        response = (
            supabase.table("messages")
            .select("role, content, created_at")
            .eq("thread_id", thread_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error("get_messages(%s) failed: %s", thread_id, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT operations
# ─────────────────────────────────────────────────────────────────────────────

def save_document(
    thread_id: str,
    filename: str,
    pinecone_namespace: Optional[str] = None,
) -> dict:
    """
    Persist document upload metadata when a PDF is ingested.

    pinecone_namespace defaults to thread_id (our naming convention).
    Returns the inserted row dict, or {} on failure.
    """
    try:
        response = (
            supabase.table("documents")
            .insert(
                {
                    "thread_id": thread_id,
                    "filename": filename,
                    "pinecone_namespace": pinecone_namespace or thread_id,
                }
            )
            .execute()
        )
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error("save_document(thread=%s, file=%s) failed: %s", thread_id, filename, e)
        return {}


def get_documents(thread_id: str) -> List[Dict[str, Any]]:
    """
    Return all documents uploaded for a thread (oldest first).

    Returns list of dicts with keys: id, filename, pinecone_namespace, upload_time.
    """
    try:
        response = (
            supabase.table("documents")
            .select("id, filename, pinecone_namespace, upload_time")
            .eq("thread_id", thread_id)
            .order("upload_time", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error("get_documents(%s) failed: %s", thread_id, e)
        return []


def document_exists(thread_id: str, filename: str) -> bool:
    """
    Return True if this filename has already been uploaded for this thread.
    Used to skip re-ingestion of the same PDF.
    """
    try:
        response = (
            supabase.table("documents")
            .select("id")
            .eq("thread_id", thread_id)
            .eq("filename", filename)
            .maybe_single()
            .execute()
        )
        return response.data is not None
    except Exception as e:
        logger.warning(
            "document_exists(thread=%s, file=%s) failed: %s", thread_id, filename, e
        )
        return False
