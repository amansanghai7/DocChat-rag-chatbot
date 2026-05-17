import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.auth import get_current_user
from app.core.exceptions import NotFoundError
from app.models.schemas import DocumentResponse, DocumentUploadResponse
from app.services import document_service, thread_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])

_ALLOWED_TYPES = {"application/pdf"}
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get(
    "/threads/{thread_id}/documents",
    response_model=List[DocumentResponse],
    summary="List uploaded documents for a thread",
)
def list_documents(thread_id: str, current_user: dict = Depends(get_current_user)):
    # Ownership check: user can only list documents in their own threads
    if not thread_service.get(thread_id, user_id=current_user["user_id"]):
        raise NotFoundError("Thread", thread_id)
    docs = document_service.list_documents(thread_id)
    return [
        DocumentResponse(
            id=d.get("id"),
            thread_id=thread_id,
            filename=d["filename"],
            pinecone_namespace=d.get("pinecone_namespace"),
            upload_time=d.get("upload_time"),
        )
        for d in docs
    ]


@router.post(
    "/threads/{thread_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a PDF and ingest it into the vector store",
)
def upload_document(
    thread_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    # Ownership check: user can only upload to their own threads
    if not thread_service.get(thread_id, user_id=current_user["user_id"]):
        raise NotFoundError("Thread", thread_id)

    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Only PDF files are accepted. Received: {file.content_type}",
        )

    file_bytes = file.file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result = document_service.upload_pdf(thread_id, file.filename, file_bytes)
    except Exception as e:
        logger.exception("Document ingestion failed for thread %s", thread_id)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if result.get("already_exists"):
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' is already uploaded for this thread.",
        )

    return DocumentUploadResponse(
        thread_id=thread_id,
        filename=file.filename or "",
        chunks=result.get("chunks", 0),
        message="Document ingested successfully.",
    )
