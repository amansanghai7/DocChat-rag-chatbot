"""
Chat endpoint tests.

chat_service.invoke_chat is mocked so tests never trigger LangGraph,
PostgreSQL, or OpenAI calls.

The chat route now checks thread ownership before invoking the AI, so tests
that reach the route body also mock thread_service.get to return a fake thread.
"""

from unittest.mock import patch
import uuid


THREAD_ID = str(uuid.uuid4())

_MOCK_THREAD = {"thread_id": THREAD_ID, "title": "Test Thread"}


def test_chat_success(client):
    with patch("app.services.thread_service.get", return_value=_MOCK_THREAD), \
         patch("app.services.chat_service.invoke_chat", return_value="Hello from AI"):
        resp = client.post(
            "/api/v1/chat",
            json={"thread_id": THREAD_ID, "message": "Hi there"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Hello from AI"
    assert body["thread_id"] == THREAD_ID


def test_chat_missing_message_field(client):
    # Pydantic rejects a body with no 'message' field → 422 (no route body executed)
    resp = client.post("/api/v1/chat", json={"thread_id": THREAD_ID})
    assert resp.status_code == 422


def test_chat_empty_message(client):
    # min_length=1 on message field → 422 (no route body executed)
    resp = client.post(
        "/api/v1/chat",
        json={"thread_id": THREAD_ID, "message": ""},
    )
    assert resp.status_code == 422


def test_chat_missing_thread_id(client):
    # Pydantic rejects missing thread_id → 422 (no route body executed)
    resp = client.post("/api/v1/chat", json={"message": "Hi"})
    assert resp.status_code == 422


def test_chat_thread_not_found_returns_404(client):
    # thread_service.get returns None → route raises NotFoundError → 404
    with patch("app.services.thread_service.get", return_value=None):
        resp = client.post(
            "/api/v1/chat",
            json={"thread_id": THREAD_ID, "message": "Hi"},
        )
    assert resp.status_code == 404


def test_chat_service_error_returns_500(client):
    with patch("app.services.thread_service.get", return_value=_MOCK_THREAD), \
         patch(
             "app.services.chat_service.invoke_chat",
             side_effect=RuntimeError("LangGraph exploded"),
         ):
        resp = client.post(
            "/api/v1/chat",
            json={"thread_id": THREAD_ID, "message": "Hi"},
        )
    assert resp.status_code == 500
    assert "Chat failed" in resp.json()["detail"]
