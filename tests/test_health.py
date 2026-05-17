"""
Health endpoint smoke tests — no mocking required, no external dependencies.
"""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_schema(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data
