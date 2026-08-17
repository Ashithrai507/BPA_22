from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from bacteria_assistant.inference import predict_bacteria_image

from ..db import get_prediction, init_db, insert_prediction, list_predictions
from ..dependencies import get_db_path
from ..models import HistoryResponse, PredictResponse

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    mode: str = Form("advanced"),
) -> PredictResponse:
    image_bytes = await image.read()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = predict_bacteria_image(image_path=tmp_path, model_path=None, mode=mode)
    finally:
        tmp_path.unlink(missing_ok=True)

    db_path = get_db_path()
    init_db(db_path)

    row_id = insert_prediction(
        db_path,
        filename=image.filename or "unknown",
        image_bytes=image_bytes,
        mode=mode,
        organism_type=result.get("organism_type"),
        predicted_species=result.get("predicted_bacteria_name"),
        gram=result.get("bacteria_type"),
        total_colonies=result.get("total_colonies_detected") or result.get("total_colonies"),
        dominant_shape=result.get("dominant_shape"),
        confidence=result.get("confidence"),
        result_json=result,
    )

    saved = get_prediction(db_path, row_id)

    return PredictResponse(
        id=row_id,
        created_at=saved["created_at"] if saved else None,
        organism_type=result.get("organism_type"),
        predicted_species=result.get("predicted_bacteria_name"),
        gram=result.get("bacteria_type"),
        total_colonies=result.get("total_colonies_detected") or result.get("total_colonies"),
        dominant_shape=result.get("dominant_shape"),
        confidence=result.get("confidence"),
        colonies=result.get("colonies"),
        morphology=result.get("final_morphology"),
    )


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = 50, offset: int = 0) -> HistoryResponse:
    db_path = get_db_path()
    init_db(db_path)
    data = list_predictions(db_path, limit=limit, offset=offset)
    return HistoryResponse(
        total=data["total"],
        items=[
            {"id": i["id"], "created_at": i["created_at"], "filename": i["filename"],
             "predicted_species": i["predicted_species"], "confidence": i["confidence"]}
            for i in data["items"]
        ],
    )


@router.get("/{prediction_id}", response_model=PredictResponse)
def get_prediction_by_id(prediction_id: int) -> PredictResponse:
    db_path = get_db_path()
    row = get_prediction(db_path, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    result = json.loads(row["result_json"])
    return PredictResponse(
        id=row["id"], created_at=row["created_at"],
        organism_type=row["organism_type"], predicted_species=row["predicted_species"],
        gram=row["gram"], total_colonies=row["total_colonies"],
        dominant_shape=row["dominant_shape"], confidence=row["confidence"],
        colonies=result.get("colonies"), morphology=result.get("final_morphology"),
    )
