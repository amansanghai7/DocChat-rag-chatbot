import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _ping_supabase() -> None:
    """Quick connectivity check — does not touch application data."""
    from supabase_client import supabase
    supabase.table("threads").select("id").limit(1).execute()


def _warmup_backend() -> None:
    """
    Eagerly import langgraph_rag_backend so the PostgreSQL connection,
    LangGraph checkpointer, and compiled graph are ready before the first
    request arrives.  Importing the module is the side effect.
    """
    import langgraph_rag_backend  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("RAG ChatBot API starting up...")

    try:
        _ping_supabase()
        logger.info("Supabase connectivity  OK")
    except Exception as exc:
        logger.warning("Supabase warmup failed (non-fatal): %s", exc)

    try:
        from app.core.auth import _get_jwks
        _get_jwks()
        logger.info("Supabase JWKS (ES256)  OK")
    except Exception as exc:
        logger.warning("JWKS warmup failed (non-fatal): %s", exc)

    try:
        _warmup_backend()
        logger.info("LangGraph backend      OK")
    except Exception as exc:
        # The backend can still load on first request; don't crash the server.
        logger.warning("LangGraph warmup failed (non-fatal): %s", exc)

    logger.info("API ready.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("RAG ChatBot API shutting down.")
