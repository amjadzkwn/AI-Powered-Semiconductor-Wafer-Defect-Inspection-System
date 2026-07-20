from datetime import date

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.db.models import Prediction

NORMAL_LABELS = {"none", "normal", "no defect", "no_defect"}


def summary(db: Session) -> dict:
    total = db.scalar(select(func.count(Prediction.id))) or 0
    defective = db.scalar(select(func.count(Prediction.id)).where(Prediction.is_defective.is_(True))) or 0
    normal = total - defective
    average_confidence = db.scalar(select(func.avg(Prediction.confidence))) or 0.0
    common = db.execute(
        select(Prediction.prediction, func.count(Prediction.id).label("count"))
        .where(Prediction.is_defective.is_(True))
        .group_by(Prediction.prediction)
        .order_by(func.count(Prediction.id).desc())
        .limit(1)
    ).first()
    labelled = db.scalar(select(func.count(Prediction.id)).where(Prediction.ground_truth.is_not(None))) or 0
    correct = db.scalar(select(func.count(Prediction.id)).where(Prediction.is_correct.is_(True))) or 0
    false_positives = db.scalar(
        select(func.count(Prediction.id)).where(
            Prediction.is_defective.is_(True),
            func.lower(Prediction.ground_truth).in_(NORMAL_LABELS),
        )
    ) or 0
    predicted_positive_labelled = db.scalar(
        select(func.count(Prediction.id)).where(
            Prediction.is_defective.is_(True), Prediction.ground_truth.is_not(None)
        )
    ) or 0
    return {
        "total_wafers": total,
        "normal_count": normal,
        "defective_count": defective,
        "defect_rate": round((defective / total * 100) if total else 0.0, 4),
        "most_common_defect": common[0] if common else None,
        "average_confidence": round(float(average_confidence), 4),
        "labelled_count": labelled,
        "estimated_false_positive_rate": (
            round(false_positives / predicted_positive_labelled * 100, 4)
            if predicted_positive_labelled else None
        ),
        "measured_accuracy": round(correct / labelled * 100, 4) if labelled else None,
    }


def daily(db: Session, days: int) -> list[dict]:
    date_expression = func.date(Prediction.created_at)
    rows = db.execute(
        select(
            date_expression.label("day"),
            func.count(Prediction.id),
            func.sum(case((Prediction.is_defective.is_(False), 1), else_=0)),
            func.sum(case((Prediction.is_defective.is_(True), 1), else_=0)),
            func.avg(Prediction.confidence),
        )
        .group_by(date_expression)
        .order_by(date_expression.desc())
        .limit(days)
    ).all()
    return [
        {
            "date": date.fromisoformat(str(row[0])),
            "total": int(row[1] or 0),
            "normal": int(row[2] or 0),
            "defective": int(row[3] or 0),
            "average_confidence": round(float(row[4] or 0), 4),
        }
        for row in reversed(rows)
    ]


def distribution(db: Session) -> list[dict]:
    total = db.scalar(select(func.count(Prediction.id))) or 0
    rows = db.execute(
        select(Prediction.prediction, func.count(Prediction.id))
        .group_by(Prediction.prediction)
        .order_by(func.count(Prediction.id).desc())
    ).all()
    return [
        {
            "defect_type": name,
            "count": count,
            "percentage": round(count / total * 100, 4) if total else 0.0,
        }
        for name, count in rows
    ]
