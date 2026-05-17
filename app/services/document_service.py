import logging
from typing import List
from db_service import get_documents, document_exists

logger = logging.getLogger(__name__)


def list_documents(thread_id: str) -> List[dict]:
    return get_documents(thread_id)


def upload_pdf(thread_id: str, filename: str, file_bytes: bytes) -> dict:
    if document_exists(thread_id, filename):
        return {"already_exists": True, "filename": filename, "chunks": 0}

    # Lazy import: ingest_pdf initialises Pinecone client and embeddings model.
    from langgraph_rag_backend import ingest_pdf

    result = ingest_pdf(file_bytes, thread_id, filename)
    return {**result, "already_exists": False}
