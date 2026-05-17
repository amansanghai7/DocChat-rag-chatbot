import logging
from typing import List, Optional
from db_service import (
    create_thread,
    get_thread,
    update_thread_title,
    get_all_threads,
    get_messages,
)

logger = logging.getLogger(__name__)


def list_threads(user_id: Optional[str] = None) -> List[dict]:
    return get_all_threads(user_id=user_id)


def create(thread_id: str, title: str = "New Chat", user_id: Optional[str] = None) -> dict:
    return create_thread(thread_id, title, user_id=user_id)


def get(thread_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    return get_thread(thread_id, user_id=user_id)


def update_title(thread_id: str, title: str) -> bool:
    return update_thread_title(thread_id, title)


def delete_completely(thread_id: str) -> dict:
    # Lazy import: avoids triggering heavy LangGraph init at service import time.
    from langgraph_rag_backend import delete_thread_completely
    return delete_thread_completely(thread_id)


def get_messages_for_thread(thread_id: str) -> List[dict]:
    return get_messages(thread_id)
