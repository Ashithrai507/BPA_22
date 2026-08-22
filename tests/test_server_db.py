from __future__ import annotations

import json
from pathlib import Path

from server.db import get_prediction, init_db, insert_prediction, list_predictions


def test_init_db_creates_file(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    assert db.exists()


def test_insert_and_get(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    row_id = insert_prediction(
        db,
        filename="test.png",
        image_bytes=b"\x89PNG",
        mode="advanced",
        organism_type="bacteria",
        predicted_species="Bacillus subtilis",
        gram="gram_positive",
        total_colonies=5,
        dominant_shape="bacilli",
        confidence=0.92,
        result_json={"predicted_bacteria_name": "Bacillus subtilis"},
    )
    assert row_id == 1
    row = get_prediction(db, row_id)
    assert row is not None
    assert row["filename"] == "test.png"
    assert row["predicted_species"] == "Bacillus subtilis"
    assert json.loads(row["result_json"])["predicted_bacteria_name"] == "Bacillus subtilis"


def test_list_predictions_pagination(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    for i in range(5):
        insert_prediction(
            db,
            filename=f"img_{i}.png",
            image_bytes=None,
            mode="basic",
            organism_type="bacteria",
            predicted_species="E. coli",
            gram="gram_negative",
            total_colonies=3,
            dominant_shape="bacilli",
            confidence=0.85,
            result_json={"i": i},
        )
    page = list_predictions(db, limit=2, offset=0)
    assert page["total"] == 5
    assert len(page["items"]) == 2
    assert page["items"][0]["filename"] == "img_4.png"  # newest first
