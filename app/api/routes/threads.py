import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.core.exceptions import NotFoundError
from app.models.schemas import (
    DeleteResponse,
    MessageResponse,
    ThreadCreateRequest,
    ThreadResponse,
    ThreadUpdateRequest,
)
from app.services import thread_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/threads", tags=["Threads"])


@router.get("", response_model=List[ThreadResponse], summary="List all threads for the current user")
def list_threads(current_user: dict = Depends(get_current_user)):
    threads = thread_service.list_threads(user_id=current_user["user_id"])
    return [
        ThreadResponse(
            thread_id=t["thread_id"],
            title=t["chat_title"],
            created_at=t.get("created_at"),
            updated_at=t.get("updated_at"),
        )
        for t in threads
    ]


@router.post("", response_model=ThreadResponse, status_code=201, summary="Create a thread")
def create_thread(body: ThreadCreateRequest, current_user: dict = Depends(get_current_user)):
    thread_service.create(body.thread_id, body.title, user_id=current_user["user_id"])
    return ThreadResponse(thread_id=body.thread_id, title=body.title)


@router.get("/{thread_id}", response_model=ThreadResponse, summary="Get a thread")
def get_thread(thread_id: str, current_user: dict = Depends(get_current_user)):
    thread = thread_service.get(thread_id, user_id=current_user["user_id"])
    if not thread:
        raise NotFoundError("Thread", thread_id)
    return ThreadResponse(**thread)


@router.patch("/{thread_id}", response_model=ThreadResponse, summary="Update thread title")
def update_thread(thread_id: str, body: ThreadUpdateRequest, current_user: dict = Depends(get_current_user)):
    if not thread_service.get(thread_id, user_id=current_user["user_id"]):
        raise NotFoundError("Thread", thread_id)
    thread_service.update_title(thread_id, body.title)
    return ThreadResponse(thread_id=thread_id, title=body.title)


@router.delete("/{thread_id}", response_model=DeleteResponse, summary="Delete a thread and all its data")
def delete_thread(thread_id: str, current_user: dict = Depends(get_current_user)):
    if not thread_service.get(thread_id, user_id=current_user["user_id"]):
        raise NotFoundError("Thread", thread_id)
    result = thread_service.delete_completely(thread_id)
    if not result.get("success"):
        errors = result.get("errors", [])
        raise HTTPException(
            status_code=500,
            detail=f"Deletion partially failed: {errors}",
        )
    return DeleteResponse(success=True, message=f"Thread {thread_id} deleted")


@router.get("/{thread_id}/messages", response_model=List[MessageResponse], summary="Get message history")
def get_messages(thread_id: str, current_user: dict = Depends(get_current_user)):
    if not thread_service.get(thread_id, user_id=current_user["user_id"]):
        raise NotFoundError("Thread", thread_id)
    messages = thread_service.get_messages_for_thread(thread_id)
    return [
        MessageResponse(
            thread_id=thread_id,
            role=m["role"],
            content=m["content"],
            created_at=m.get("created_at"),
        )
        for m in messages
    ]
