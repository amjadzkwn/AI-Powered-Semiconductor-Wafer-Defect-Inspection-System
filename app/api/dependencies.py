from functools import lru_cache

from app.ml.inference import ModelService


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
