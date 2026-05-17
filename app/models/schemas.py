from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    thread_id: str = Field(..., description="UUID of the conversation thread")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    thread_id: str = Field(..., description="UUID of the conversation thread")


class HealthResponse(BaseModel):
    status: str
    version: str


class ThreadCreateRequest(BaseModel):
    thread_id: str = Field(..., description="UUID of the thread to create")
    title: str = Field("New Chat", description="Display title for the thread")


class ThreadResponse(BaseModel):
    thread_id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageResponse(BaseModel):
    id: Optional[str] = None
    thread_id: str
    role: str
    content: str
    created_at: Optional[str] = None


class DocumentResponse(BaseModel):
    id: Optional[str] = None
    thread_id: str
    filename: str
    pinecone_namespace: Optional[str] = None
    upload_time: Optional[str] = None


class ThreadUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="New display title for the thread")


class DeleteResponse(BaseModel):
    success: bool
    message: str


class DocumentUploadResponse(BaseModel):
    thread_id: str
    filename: str
    chunks: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
