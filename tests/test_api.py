"""FastAPI service tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
HAS_HTTPX = importlib.util.find_spec("httpx") is not None


@pytest.mark.skipif(not (HAS_FASTAPI and HAS_HTTPX), reason="fastapi + httpx required")
def test_health_and_formats():
    from fastapi.testclient import TestClient
    from ekg.api.server import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/formats")
    assert r.status_code == 200
    assert "supported" in r.json()


@pytest.mark.skipif(not (HAS_FASTAPI and HAS_HTTPX), reason="fastapi + httpx required")
def test_models_endpoint_empty_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EKG_RUNS_DIR", str(tmp_path))
    import importlib
    import ekg.api.server as server
    importlib.reload(server)

    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()["models"] == {}


@pytest.mark.skipif(not (HAS_FASTAPI and HAS_HTTPX), reason="fastapi + httpx required")
def test_classify_endpoint_503_when_no_models(tmp_path, monkeypatch):
    monkeypatch.setenv("EKG_RUNS_DIR", str(tmp_path))
    import importlib
    import ekg.api.server as server
    importlib.reload(server)

    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("# sampling_rate,360\nlead_I\n0.1\n0.2\n0.3\n")
    with csv_path.open("rb") as f:
        r = client.post("/classify", files={"file": ("dummy.csv", f, "text/csv")})
    assert r.status_code == 503
