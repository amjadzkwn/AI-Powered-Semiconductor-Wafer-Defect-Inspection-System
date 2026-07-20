from fastapi import APIRouter, Depends

from app.api.dependencies import get_model_service
from app.ml.inference import ModelService

router = APIRouter(tags=["system"])


@router.get("/model-info")
def model_info(model_service: ModelService = Depends(get_model_service)):
    return model_service.model_info
