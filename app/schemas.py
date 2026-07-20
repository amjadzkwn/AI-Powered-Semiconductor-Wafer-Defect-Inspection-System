from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    id: int
    prediction: str
    confidence: float
    is_defective: bool
    heatmap: str
    probabilities: dict[str, float]
    inference_ms: float
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    prediction: str
    confidence: float
    is_defective: bool
    ground_truth: str | None
    is_correct: bool | None
    inference_ms: float
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[PredictionHistoryItem]


class GroundTruthUpdate(BaseModel):
    ground_truth: str = Field(min_length=1, max_length=100)


class AnalyticsSummary(BaseModel):
    total_wafers: int
    normal_count: int
    defective_count: int
    defect_rate: float
    most_common_defect: str | None
    average_confidence: float
    labelled_count: int
    estimated_false_positive_rate: float | None
    measured_accuracy: float | None


class DailyMetric(BaseModel):
    date: date
    total: int
    normal: int
    defective: int
    average_confidence: float


class DistributionMetric(BaseModel):
    defect_type: str
    count: int
    percentage: float
