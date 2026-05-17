import logging
from db_service import thread_exists, save_message

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(block) for block in content)
    return str(content)


def invoke_chat(thread_id: str, message: str) -> str:
    # Lazy import: langgraph_rag_backend has heavy module-level side effects
    # (PostgreSQL connection, checkpointer setup, graph compilation).
    # Deferring until first actual call keeps uvicorn startup instant.
    from langgraph_rag_backend import chatbot, create_thread_metadata

    if not thread_exists(thread_id):
        create_thread_metadata(thread_id)

    save_message(thread_id, "user", message)

    result = chatbot.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    response_text = _extract_text(result["messages"][-1].content)
    save_message(thread_id, "assistant", response_text)
    return response_text
