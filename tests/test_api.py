import io
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from otasentry.api import app  # noqa: E402


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_rejects_bad_archives():
    c = TestClient(app)
    assert c.post("/v1/campaigns/analyze", files={"archive": ("x.zip", b"nope")}).status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.log", "x")
    assert c.post("/v1/campaigns/analyze", files={"archive": ("x.zip", buf.getvalue())}).status_code == 400


def test_analyze_archive(faulty_campaign):
    camp, _ = faulty_campaign
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in camp.rglob("*.log"):
            z.write(p, p.relative_to(camp.parent))
    r = TestClient(app).post("/v1/campaigns/analyze", files={"archive": ("c.zip", buf.getvalue())})
    assert r.status_code == 200
    assert r.json()["gate"]["decision"] == "NO-GO"
