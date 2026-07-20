from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    heatmap_path: Mapped[str] = mapped_column(String(500), nullable=False)
    prediction: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_defective: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    ground_truth: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
