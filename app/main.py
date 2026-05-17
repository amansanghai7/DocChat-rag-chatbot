"""
FastAPI entry point for RAG ChatBot API.

Run from the project root directory:
    uvicorn app.main:app --reload --port 8000

Streamlit frontend continues to run independently:
    streamlit run frontend_rag.py

Both share the same backend modules (langgraph_rag_backend, db_service).
"""

import logging
import os
import sys

# Ensure project root is on sys.path so root-level modules are importable
# when uvicorn is invoked from a different working directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health, threads
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.middleware import RequestLoggingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "REST API for the RAG ChatBot. "
        "Provides chat, thread, and document management endpoints. "
        "Backed by LangGraph, Pinecone, and Supabase."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Order matters: RequestLoggingMiddleware wraps CORSMiddleware so every
# request is logged including pre-flight OPTIONS calls.
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(threads.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
