from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "WM-811K Inspection API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./wm811k.db"
    model_path: Path = Path("artifacts/custom_cnn_best.pt")
    label_mapping_path: Path = Path("artifacts/label_mapping.json")
    upload_dir: Path = Path("data/uploads")
    heatmap_dir: Path = Path("data/heatmaps")
    max_upload_mb: int = 10
    cors_origins: list[str] = ["http://localhost:8501"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.heatmap_dir.mkdir(parents=True, exist_ok=True)
    return settings
