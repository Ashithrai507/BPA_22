from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)

_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
FAKE_RESULT = {
    "organism_type": "bacteria",
    "predicted_bacteria_name": "E. coli",
    "bacteria_type": "gram_negative",
    "total_colonies_detected": 8,
    "dominant_shape": "bacilli",
    "confidence": 0.88,
}


def test_health() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_species_list() -> None:
    resp = client.get("/api/species")
    assert resp.status_code == 200
    assert len(resp.json()) == 10


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_then_fetch_by_id(mock_predict) -> None:
    post_resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    assert post_resp.status_code == 200
    pred_id = post_resp.json()["id"]
    get_resp = client.get(f"/api/predict/{pred_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["predicted_species"] == "E. coli"
    hist_resp = client.get("/api/predict/history?limit=10")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["total"] >= 1
