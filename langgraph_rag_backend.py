from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict, cast

import psycopg
from psycopg.rows import DictRow, dict_row
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Supabase DB service (application-level tables)
import db_service

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM + Embeddings
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pinecone Setup  (unchanged — Pinecone is still the vector store)
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Failed to connect to Pinecone index '{PINECONE_INDEX_NAME}': {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# 3. Thread-specific retriever cache and in-session document metadata
# ─────────────────────────────────────────────────────────────────────────────
_THREAD_RETRIEVERS: Dict[str, Any] = {}   # Pinecone retriever cache (per session)
_THREAD_METADATA: Dict[str, dict] = {}    # Chunk/page counts (in-memory, display only)


def _generate_chunk_id(thread_id: str, filename: str, page: int, chunk_idx: int) -> str:
    """
    Generate a deterministic Pinecone vector ID.
    Format: {thread_id}:{safe_filename}:p{page}:c{chunk_idx}
    Prevents duplicate vectors on re-upload.
    """
    safe_filename = filename.replace(" ", "_").replace(":", "_").replace("/", "_")
    return f"{thread_id}:{safe_filename}:p{page}:c{chunk_idx}"


def _get_retriever(thread_id: Optional[str]):
    """
    Return a cached Pinecone retriever for the thread, or create one if the
    namespace exists.  Returns None if the thread has no indexed documents.
    """
    if not thread_id:
        return None

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if thread_id in namespaces and namespaces[thread_id]["vector_count"] > 0:
            vectorstore = PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings,
                namespace=thread_id,
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )
            _THREAD_RETRIEVERS[thread_id] = retriever
            return retriever
    except Exception as e:
        logger.warning("Error checking Pinecone namespace for thread %s: %s", thread_id, e)

    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingest a PDF into Pinecone (namespace = thread_id) and persist document
    metadata to Supabase.

    Steps:
      1. Write bytes to a temp file for PyPDFLoader
      2. Split into chunks with deterministic IDs
      3. Batch-upsert to Pinecone
      4. Cache retriever for immediate use
      5. Save document metadata to Supabase documents table

    Returns a summary dict used by the frontend for display.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        upload_timestamp = datetime.utcnow().isoformat()
        filename_to_use = filename or os.path.basename(temp_path)

        texts, metadatas, ids = [], [], []
        for chunk_idx, chunk in enumerate(chunks):
            page_num = chunk.metadata.get("page", 0)
            chunk_id = _generate_chunk_id(
                thread_id=str(thread_id),
                filename=filename_to_use,
                page=page_num,
                chunk_idx=chunk_idx,
            )
            texts.append(chunk.page_content)
            metadatas.append(
                {
                    "thread_id": str(thread_id),
                    "filename": filename_to_use,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "upload_timestamp": upload_timestamp,
                    "source_type": "pdf",
                    "text": chunk.page_content,
                }
            )
            ids.append(chunk_id)

        # Batch-upsert to Pinecone (100 chunks per batch)
        batch_size = 100
        total_upserted = 0
        print(f"📤 Upserting {len(texts)} chunks to Pinecone (namespace: {thread_id})...")

        for i in range(0, len(texts), batch_size):
            vectorstore = PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings,
                namespace=str(thread_id),
            )
            vectorstore.add_texts(
                texts=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )
            total_upserted += len(texts[i : i + batch_size])
            print(f"  ✅ Batch {i // batch_size + 1}: {len(texts[i:i+batch_size])} chunks upserted")

        print(f"✅ Total upserted: {total_upserted} chunks")

        # Cache retriever for immediate use in this session
        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=str(thread_id),
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        # Keep chunk/page counts in memory for display (not stored in Supabase)
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename_to_use,
            "documents": len(docs),
            "chunks": len(chunks),
            "upload_timestamp": upload_timestamp,
        }

        # Persist document metadata to Supabase (non-critical — won't break ingestion)
        try:
            db_service.save_document(
                thread_id=str(thread_id),
                filename=filename_to_use,
                pinecone_namespace=str(thread_id),
            )
        except Exception as e:
            logger.warning("Could not save document metadata to Supabase: %s", e)

        return {
            "filename": filename_to_use,
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tools
# ─────────────────────────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Uses Pinecone with namespace-based thread isolation.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    try:
        result = retriever.invoke(query)
        return {
            "query": query,
            "context": [doc.page_content for doc in result],
            "metadata": [doc.metadata for doc in result],
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
            "num_results": len(result),
        }
    except Exception as e:
        return {"error": f"Retrieval failed: {str(e)}", "query": query}


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────────────────────────────────────────
# 5. LangGraph State
# ─────────────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────
def chat_node(state: ChatState, config=None):
    """LLM node — answers the user or decides to call a tool."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`. "
            "The system uses Pinecone for persistent vector storage. "
            "You can also use the web search, stock price, and calculator tools when helpful. "
            "If no document is available, ask the user to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PostgreSQL Checkpointer  (replaces SqliteSaver)
#
# LangGraph manages its own tables (checkpoints, checkpoint_blobs,
# checkpoint_writes, checkpoint_migrations) inside the same Supabase
# PostgreSQL database.  These are separate from our application tables.
#
# DATABASE_URL must be the direct PostgreSQL connection string:
#   postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
# Find it in: Supabase Dashboard → Settings → Database → Connection string
#
# prepare_threshold=0 disables server-side prepared statements — required
# when connecting through Supabase's PgBouncer pooler (safe to keep for
# direct connections too).
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is missing from environment variables. "
        "Add the direct PostgreSQL connection string from Supabase → Settings → Database."
    )

try:
    _raw = psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=0)
    _raw.row_factory = dict_row  # type: ignore[assignment]
    _pg_conn = cast(psycopg.Connection[DictRow], _raw)
    checkpointer = PostgresSaver(_pg_conn)
    checkpointer.setup()  # Creates LangGraph checkpoint tables if they don't exist yet
    print("✅ Connected to PostgreSQL (Supabase) for LangGraph checkpoints")
except Exception as e:
    print(f"❌ Failed to connect to PostgreSQL for checkpoints: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# 8. Compile Graph
# ─────────────────────────────────────────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Thread metadata helpers  (thin wrappers — delegate to db_service)
#
# These keep the same public API so frontend_rag.py needs minimal changes.
# ─────────────────────────────────────────────────────────────────────────────

def create_thread_metadata(thread_id: str, title: str = "New Chat", user_id: Optional[str] = None) -> None:
    """Create a thread row in Supabase.  Safe to call multiple times."""
    db_service.create_thread(thread_id, title, user_id=user_id)


def update_thread_title(thread_id: str, title: str) -> None:
    """Update the display title for a thread."""
    db_service.update_thread_title(thread_id, title)


def get_thread_title(thread_id: str) -> str:
    """Return the display title for a thread, defaulting to 'New Chat'."""
    return db_service.get_thread_title(thread_id)


def get_all_threads_with_metadata(user_id: Optional[str] = None) -> List[dict]:
    """
    Return all threads ordered by last activity (updated_at DESC).
    Source: Supabase threads table (single query, no checkpoint scan).
    When user_id is provided only that user's threads are returned.
    """
    return db_service.get_all_threads(user_id=user_id)


def generate_chat_title(first_message: str) -> str:
    """
    Ask the LLM to generate a concise ChatGPT-style title (3–6 words)
    from the first user message.  Falls back to a word-truncated version.
    """
    try:
        title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = (
            f'Generate a short, concise title (3-6 words max) for a conversation '
            f'that starts with this message:\n\n"{first_message}"\n\n'
            "Rules:\n"
            "- Maximum 6 words\n"
            "- No quotes or punctuation at the end\n"
            "- Capture the main topic\n"
            "- Be specific and descriptive\n"
            '- Examples: "Python Factorial Code", "Benefits of RAG Systems"\n\n'
            "Title:"
        )
        response = title_llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")

        if not title or len(title) > 60:
            words = first_message.split()[:6]
            title = " ".join(words)
            if len(first_message.split()) > 6:
                title += "..."

        return title[:60]
    except Exception:
        words = first_message.split()[:6]
        title = " ".join(words)
        if len(first_message.split()) > 6:
            title += "..."
        return title[:60]


def retrieve_all_threads() -> list:
    """Legacy helper — returns thread IDs from LangGraph checkpoints."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            tid = (checkpoint.config.get("configurable") or {}).get("thread_id")
            if tid:
                all_threads.add(str(tid))
    except Exception as e:
        logger.warning("Could not list LangGraph checkpoints: %s", e)
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """Return True if the thread has vectors in Pinecone."""
    if str(thread_id) in _THREAD_RETRIEVERS:
        return True
    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return thread_id in namespaces and namespaces[thread_id]["vector_count"] > 0
    except Exception as e:
        logger.warning("Error checking Pinecone for thread %s: %s", thread_id, e)
        return False


def thread_document_metadata(thread_id: str) -> dict:
    """
    Return display metadata for the most recently uploaded document.

    Checks in-memory cache first (has chunk/page counts from this session).
    Falls back to Supabase documents table (has filename, survives restart).
    """
    if str(thread_id) in _THREAD_METADATA:
        return _THREAD_METADATA[str(thread_id)]

    docs = db_service.get_documents(str(thread_id))
    if docs:
        latest = docs[-1]
        return {"filename": latest["filename"]}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Pinecone management utilities  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_pinecone_stats() -> dict:
    """Return Pinecone index statistics."""
    try:
        stats = pinecone_index.describe_index_stats()
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {}),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0.0),
        }
    except Exception as e:
        return {"error": str(e)}


def delete_thread_vectors(thread_id: str) -> dict:
    """Delete all Pinecone vectors for a thread namespace."""
    try:
        pinecone_index.delete(delete_all=True, namespace=thread_id)
        _THREAD_RETRIEVERS.pop(thread_id, None)
        _THREAD_METADATA.pop(thread_id, None)
        print(f"✅ Deleted Pinecone vectors for thread: {thread_id}")
        return {"success": True, "thread_id": thread_id}
    except Exception as e:
        print(f"❌ Failed to delete vectors for thread {thread_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_thread_completely(thread_id: str) -> dict:
    """
    Fully delete a thread and all its data:

      1. Supabase threads row  (CASCADE removes messages + documents rows)
      2. LangGraph checkpoint rows in PostgreSQL  (checkpoints, blobs, writes)
      3. Pinecone namespace (vectors)
      4. In-memory retriever + metadata cache

    Returns a result dict with per-step flags so the frontend can give
    meaningful feedback.
    """
    results: dict = {
        "success": True,
        "thread_id": thread_id,
        "deleted": {
            "metadata": False,
            "checkpoints": False,
            "pinecone_vectors": False,
            "cache": False,
        },
        "errors": [],
    }

    # 1. Delete from Supabase  (cascade handles messages + documents)
    try:
        db_service.delete_thread(thread_id)
        results["deleted"]["metadata"] = True
        print(f"✅ Deleted Supabase data for thread: {thread_id}")
    except Exception as e:
        msg = f"Failed to delete Supabase thread data: {e}"
        results["errors"].append(msg)
        logger.error(msg)

    # 2. Delete LangGraph checkpoint rows from PostgreSQL
    #    LangGraph uses thread_id as TEXT across three tables.
    try:
        with _pg_conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoints        WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM checkpoint_blobs   WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM checkpoint_writes  WHERE thread_id = %s", (thread_id,))
        results["deleted"]["checkpoints"] = True
        print(f"✅ Deleted LangGraph checkpoints for thread: {thread_id}")
    except Exception as e:
        msg = f"Failed to delete LangGraph checkpoints: {e}"
        results["errors"].append(msg)
        logger.error(msg)

    # 3. Delete Pinecone vectors (only if the namespace exists)
    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if thread_id in namespaces and namespaces[thread_id].get("vector_count", 0) > 0:
            pinecone_index.delete(delete_all=True, namespace=thread_id)
            print(f"✅ Deleted Pinecone vectors for thread: {thread_id}")
        else:
            print(f"ℹ️  No Pinecone vectors found for thread: {thread_id} (skipped)")

        results["deleted"]["pinecone_vectors"] = True
    except Exception as e:
        err_str = str(e).lower()
        if any(x in err_str for x in ("not found", "404", "namespace")):
            # Namespace never existed — not a real error
            results["deleted"]["pinecone_vectors"] = True
            print(f"ℹ️  Pinecone namespace already gone for thread: {thread_id}")
        else:
            msg = f"Failed to delete Pinecone vectors: {e}"
            results["errors"].append(msg)
            logger.error(msg)

    # 4. Clear in-memory caches
    try:
        _THREAD_RETRIEVERS.pop(thread_id, None)
        _THREAD_METADATA.pop(thread_id, None)
        results["deleted"]["cache"] = True
        print(f"✅ Cleared in-memory cache for thread: {thread_id}")
    except Exception as e:
        msg = f"Failed to clear cache: {e}"
        results["errors"].append(msg)
        logger.error(msg)

    if results["errors"]:
        results["success"] = False
        print(f"⚠️  Thread deletion completed with errors: {thread_id}")
    else:
        print(f"🎉 Thread completely deleted: {thread_id}")

    return results


def list_thread_namespaces() -> List[str]:
    """Return all Pinecone namespace names (one per thread with documents)."""
    try:
        stats = pinecone_index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())
    except Exception as e:
        logger.error("Failed to list Pinecone namespaces: %s", e)
        return []


def get_thread_vector_count(thread_id: str) -> int:
    """Return the number of vectors stored for a specific thread namespace."""
    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return namespaces.get(thread_id, {}).get("vector_count", 0)
    except Exception as e:
        logger.warning("Error getting vector count for thread %s: %s", thread_id, e)
        return 0
