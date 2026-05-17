"""
Shared pytest fixtures.

The session-scoped `client` fixture does two things:
  1. Overrides the get_current_user dependency so every route receives a
     fake user without needing a real JWT.  This keeps all existing tests
     working without credentials.
  2. Returns a TestClient that raises server-side exceptions (500s become
     Python exceptions so test failures are obvious).

Service-layer calls are mocked per test — tests never hit real Supabase or
LangGraph, only the HTTP routing and Pydantic validation layers are exercised.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.auth import get_current_user
from app.main import app

MOCK_USER = {
    "user_id": "00000000-0000-0000-0000-000000000001",
    "email": "test@example.com",
}


@pytest.fixture(scope="session")
def client() -> TestClient:
    # Replace the real JWT dependency with a lambda that always returns the
    # mock user.  Any route that calls Depends(get_current_user) gets MOCK_USER
    # instead of attempting JWT verification.
    app.dependency_overrides[get_current_user] = lambda: MOCK_USER
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()
