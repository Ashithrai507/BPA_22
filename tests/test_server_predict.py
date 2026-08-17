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

FAKE_RESULT = {
    "organism_type": "bacteria",
    "predicted_bacteria_name": "Bacillus subtilis",
    "bacteria_type": "gram_positive",
    "total_colonies_detected": 5,
    "dominant_shape": "bacilli",
    "confidence": 0.95,
}

_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_returns_result(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["organism_type"] == "bacteria"
    assert body["predicted_species"] == "Bacillus subtilis"
    assert body["confidence"] == 0.95


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_saves_to_history(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    pred_id = resp.json()["id"]
    hist = client.get("/api/predict/history")
    assert hist.status_code == 200
    assert any(item["id"] == pred_id for item in hist.json()["items"])


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_get_prediction_by_id(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "advanced"})
    pred_id = resp.json()["id"]
    detail = client.get(f"/api/predict/{pred_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == pred_id
