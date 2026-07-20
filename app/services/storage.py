from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings


def save_binary(content: bytes, original_name: str, destination: Path) -> Path:
    suffix = Path(original_name).suffix.lower() or ".bin"
    path = destination / f"{uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def save_upload(content: bytes, original_name: str) -> Path:
    return save_binary(content, original_name, get_settings().upload_dir)


def save_heatmap(content: bytes) -> Path:
    return save_binary(content, "heatmap.png", get_settings().heatmap_dir)
