from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_model_service
from app.core.config import get_settings
from app.db.models import Prediction
from app.db.session import get_db
from app.ml.inference import ModelService
from app.ml.preprocessing import InvalidWaferInput, decode_upload
from app.schemas import (
    GroundTruthUpdate, PredictionHistoryItem, PredictionHistoryResponse, PredictionResponse,
)
from app.services.storage import save_heatmap, save_upload

router = APIRouter(tags=["predictions"])


@router.post("/upload-image", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    model_service: ModelService = Depends(get_model_service),
):
    settings = get_settings()
    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Maximum upload size is {settings.max_upload_mb} MB.")

    filename = file.filename or "wafer.png"
    try:
        wafer_map = decode_upload(content, filename)
        result = model_service.predict(wafer_map)
    except InvalidWaferInput as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    upload_path = save_upload(content, filename)
    heatmap_path = save_heatmap(result.heatmap_bytes)
    row = Prediction(
        filename=Path(filename).name,
        stored_image_path=str(upload_path),
        heatmap_path=str(heatmap_path),
        prediction=result.prediction,
        confidence=result.confidence,
        is_defective=result.is_defective,
        probabilities=result.probabilities,
        inference_ms=result.inference_ms,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return PredictionResponse(
        id=row.id,
        prediction=row.prediction,
        confidence=row.confidence,
        is_defective=row.is_defective,
        heatmap=result.heatmap_data_uri,
        probabilities=row.probabilities,
        inference_ms=row.inference_ms,
        created_at=row.created_at,
    )


@router.get("/predictions", response_model=PredictionHistoryResponse)
def prediction_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    prediction: str | None = None,
    defective_only: bool = False,
    db: Session = Depends(get_db),
):
    filters = []
    if prediction:
        filters.append(Prediction.prediction == prediction)
    if defective_only:
        filters.append(Prediction.is_defective.is_(True))
    total = db.scalar(select(func.count(Prediction.id)).where(*filters)) or 0
    items = db.scalars(
        select(Prediction).where(*filters).order_by(Prediction.created_at.desc())
        .offset((page - 1) * page_size).limit(page_size)
    ).all()
    return PredictionHistoryResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem)
def prediction_detail(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return row


@router.patch("/predictions/{prediction_id}/ground-truth", response_model=PredictionHistoryItem)
def update_ground_truth(
    prediction_id: int,
    payload: GroundTruthUpdate,
    db: Session = Depends(get_db),
    model_service: ModelService = Depends(get_model_service),
):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    canonical = {name.lower(): name for name in model_service.class_names}
    supplied = payload.ground_truth.strip()
    if supplied.lower() not in canonical:
        raise HTTPException(status_code=422, detail={"message": "Unknown class", "allowed": model_service.class_names})
    row.ground_truth = canonical[supplied.lower()]
    row.is_correct = row.prediction.lower() == row.ground_truth.lower()
    db.commit()
    db.refresh(row)
    return row
