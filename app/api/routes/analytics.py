from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import AnalyticsSummary, DailyMetric, DistributionMetric
from app.services import analytics as analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
def analytics_summary(db: Session = Depends(get_db)):
    return analytics_service.summary(db)


@router.get("/daily", response_model=list[DailyMetric])
def daily_analytics(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    return analytics_service.daily(db, days)


@router.get("/distribution", response_model=list[DistributionMetric])
def defect_distribution(db: Session = Depends(get_db)):
    return analytics_service.distribution(db)
