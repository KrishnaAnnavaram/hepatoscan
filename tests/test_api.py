"""HTTP tests. They need fastapi and httpx (both in the dev extra)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from hepatoscan.assistant.chat import Assistant  # noqa: E402
from hepatoscan.service.api import create_app  # noqa: E402
from hepatoscan.service.audit import AuditLog  # noqa: E402
from hepatoscan.service.handlers import Service  # noqa: E402
from hepatoscan.volume import save_npz  # noqa: E402


@pytest.fixture
def client(fitted_baseline):
    svc = Service(fitted_baseline, Assistant(), AuditLog(None), "secret", 2 * 1024 * 1024)
    return TestClient(create_app(svc))


AUTH = {"Authorization": "Bearer secret"}


def test_health_page_and_headers(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    page = client.get("/")
    js = client.get("/app.js").text
    assert "Not a diagnosis" in page.text
    assert "localStorage." not in js and "sessionStorage." not in js and "innerHTML" not in js
    assert client.get("/docs").status_code == 404


def test_segment_requires_token(client):
    assert client.post("/segment", content=b"x", headers={"X-Filename": "a.npz"}).status_code == 401


def test_segment_ok_and_rejections(client, phantoms, tmp_path):
    data = save_npz(phantoms[7], tmp_path / "v.npz").read_bytes()
    r = client.post("/segment", content=data, headers={**AUTH, "X-Filename": "v.npz"})
    assert r.status_code == 200 and r.json()["summary"]["liver_volume_ml"] > 0
    assert client.post("/segment", content=b"\x89PNG....", headers={**AUTH, "X-Filename": "a.png"}).status_code == 422
    big = b"0" * (2 * 1024 * 1024 + 1)
    assert client.post("/segment", content=big, headers={**AUTH, "X-Filename": "a.npz"}).status_code == 413


def test_chat_flow(client):
    r = client.post("/chat", json={"question": "What does the lesion count mean?"}, headers=AUTH)
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert client.delete(f"/chat/{sid}", headers=AUTH).json() == {"deleted": True}
    assert client.post("/chat", content=b"not json", headers=AUTH).status_code == 422
    assert client.post("/chat", json={"question": ""}, headers=AUTH).status_code == 422


def test_service_without_token_refuses(fitted_baseline):
    svc = Service(fitted_baseline, Assistant(), AuditLog(None), None, 1024)
    c = TestClient(create_app(svc))
    assert c.post("/chat", json={"question": "hi"}, headers=AUTH).status_code == 503
