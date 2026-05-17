"""
Thread endpoint tests.

All db / LangGraph calls are mocked at the service layer so tests run
without real Supabase credentials and without network access.
"""

from unittest.mock import patch
import uuid

MOCK_THREAD = {
    "thread_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "title": "Test Thread",
    "created_at": "2024-01-01T00:00:00+00:00",
    "updated_at": "2024-01-01T00:00:00+00:00",
}


# ── List ─────────────────────────────────────────────────────────────────────

def test_list_threads_empty(client):
    with patch("app.services.thread_service.list_threads", return_value=[]):
        resp = client.get("/api/v1/threads")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_threads_returns_data(client):
    raw = [{"thread_id": MOCK_THREAD["thread_id"], "chat_title": "Test Thread",
            "created_at": MOCK_THREAD["created_at"], "updated_at": MOCK_THREAD["updated_at"]}]
    with patch("app.services.thread_service.list_threads", return_value=raw):
        resp = client.get("/api/v1/threads")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["thread_id"] == MOCK_THREAD["thread_id"]
    assert body[0]["title"] == "Test Thread"


# ── Create ───────────────────────────────────────────────────────────────────

def test_create_thread_returns_201(client):
    tid = str(uuid.uuid4())
    with patch("app.services.thread_service.create", return_value={}):
        resp = client.post("/api/v1/threads", json={"thread_id": tid, "title": "New Chat"})
    assert resp.status_code == 201
    assert resp.json()["thread_id"] == tid


def test_create_thread_default_title(client):
    tid = str(uuid.uuid4())
    with patch("app.services.thread_service.create", return_value={}):
        resp = client.post("/api/v1/threads", json={"thread_id": tid})
    assert resp.status_code == 201
    assert resp.json()["title"] == "New Chat"


# ── Get ──────────────────────────────────────────────────────────────────────

def test_get_thread_not_found(client):
    with patch("app.services.thread_service.get", return_value=None):
        resp = client.get(f"/api/v1/threads/{MOCK_THREAD['thread_id']}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_get_thread_found(client):
    with patch("app.services.thread_service.get", return_value=MOCK_THREAD):
        resp = client.get(f"/api/v1/threads/{MOCK_THREAD['thread_id']}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Test Thread"


# ── Update ───────────────────────────────────────────────────────────────────

def test_update_thread_not_found(client):
    with patch("app.services.thread_service.get", return_value=None):
        resp = client.patch(
            f"/api/v1/threads/{MOCK_THREAD['thread_id']}",
            json={"title": "New Title"},
        )
    assert resp.status_code == 404


def test_update_thread_success(client):
    with patch("app.services.thread_service.get", return_value=MOCK_THREAD), \
         patch("app.services.thread_service.update_title", return_value=True):
        resp = client.patch(
            f"/api/v1/threads/{MOCK_THREAD['thread_id']}",
            json={"title": "Renamed"},
        )
    assert resp.status_code == 200
    assert resp.json()["title"] == "Renamed"


# ── Delete ───────────────────────────────────────────────────────────────────

def test_delete_thread_not_found(client):
    with patch("app.services.thread_service.get", return_value=None):
        resp = client.delete(f"/api/v1/threads/{MOCK_THREAD['thread_id']}")
    assert resp.status_code == 404


def test_delete_thread_success(client):
    delete_result = {"success": True, "thread_id": MOCK_THREAD["thread_id"], "errors": []}
    with patch("app.services.thread_service.get", return_value=MOCK_THREAD), \
         patch("app.services.thread_service.delete_completely", return_value=delete_result):
        resp = client.delete(f"/api/v1/threads/{MOCK_THREAD['thread_id']}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ── Messages ─────────────────────────────────────────────────────────────────

def test_get_messages_not_found(client):
    with patch("app.services.thread_service.get", return_value=None):
        resp = client.get(f"/api/v1/threads/{MOCK_THREAD['thread_id']}/messages")
    assert resp.status_code == 404


def test_get_messages_returns_list(client):
    msgs = [
        {"role": "user", "content": "Hello", "created_at": "2024-01-01T00:00:00+00:00"},
        {"role": "assistant", "content": "Hi!", "created_at": "2024-01-01T00:00:01+00:00"},
    ]
    with patch("app.services.thread_service.get", return_value=MOCK_THREAD), \
         patch("app.services.thread_service.get_messages_for_thread", return_value=msgs):
        resp = client.get(f"/api/v1/threads/{MOCK_THREAD['thread_id']}/messages")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["role"] == "user"
    assert body[1]["role"] == "assistant"
