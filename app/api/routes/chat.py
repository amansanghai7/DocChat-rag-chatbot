import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.core.exceptions import NotFoundError
from app.models.schemas import ChatRequest, ChatResponse
from app.services import chat_service, thread_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send a message and get a response")
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]

    # Verify the thread exists and belongs to this user before invoking the AI.
    # This prevents one user from reading another user's conversation history
    # through the LangGraph checkpointer.
    if not thread_service.get(request.thread_id, user_id=user_id):
        raise NotFoundError("Thread", request.thread_id)

    try:
        response_text = chat_service.invoke_chat(request.thread_id, request.message)
        return ChatResponse(response=response_text, thread_id=request.thread_id)
    except Exception as e:
        logger.exception("Chat failed for thread %s", request.thread_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
