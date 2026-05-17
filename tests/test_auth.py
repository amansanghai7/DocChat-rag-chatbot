"""
Authentication tests.

These tests exercise the real get_current_user dependency (not the mock
override from conftest.py).  A separate `raw_client` fixture is used that
does NOT inject a fake user, so the routes actually attempt JWT verification.

We use raise_server_exceptions=False here so HTTPExceptions (401, 403) come
back as proper HTTP responses rather than bubbling up as Python exceptions.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.auth import get_current_user
from app.main import app


@pytest.fixture
def raw_client():
    """
    TestClient with no auth override.
    Temporarily removes the session-scoped mock so real auth runs.
    """
    original = app.dependency_overrides.pop(get_current_user, None)
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    # Restore the mock override so other tests continue to work
    if original is not None:
        app.dependency_overrides[get_current_user] = original


# ── Missing token ─────────────────────────────────────────────────────────────

def test_no_token_returns_401(raw_client):
    # No Authorization header at all
    resp = raw_client.get("/api/v1/threads")
    assert resp.status_code == 401
    assert "Missing" in resp.json()["detail"]


def test_no_token_on_chat_returns_401(raw_client):
    resp = raw_client.post("/api/v1/chat", json={"thread_id": "abc", "message": "hi"})
    assert resp.status_code == 401


# ── Invalid / malformed token ─────────────────────────────────────────────────

def test_invalid_token_returns_401(raw_client):
    # A structurally malformed token fails at JWT header parsing (DecodeError)
    # before any JWKS network call is made — no mocking required.
    resp = raw_client.get(
        "/api/v1/threads",
        headers={"Authorization": "Bearer this.is.not.a.real.jwt"},
    )
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


def test_malformed_bearer_returns_401(raw_client):
    # Completely wrong format (no token after Bearer)
    resp = raw_client.get(
        "/api/v1/threads",
        headers={"Authorization": "Bearer "},
    )
    assert resp.status_code == 401


# ── Public routes are unaffected ──────────────────────────────────────────────

def test_health_is_public(raw_client):
    # /health must remain accessible without a token
    resp = raw_client.get("/health")
    assert resp.status_code == 200
