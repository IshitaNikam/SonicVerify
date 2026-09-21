"""Phone-number lookup and community reporting."""
import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import FraudReport, PhoneNumber, utcnow
from app.schemas.numbers import NumberIntelligence, NumberLookupResponse, ReportResponse
from app.services.risk_service import get_number_risk_level
from app.utils.validation import normalize_phone_number

REPORTED_MESSAGE = (
    "Community-reported suspicious number. Community reports are an indicator, not proof of fraud."
)
UNREPORTED_MESSAGE = "No community reports found for this number. This does not guarantee the number is safe."


def _load_categories(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        return [str(v) for v in value] if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []


def _get_record(db: Session, phone: str) -> PhoneNumber | None:
    return db.execute(select(PhoneNumber).where(PhoneNumber.phone_number == phone)).scalar_one_or_none()


def _to_intelligence(phone: str, record: PhoneNumber | None) -> NumberIntelligence:
    if record is None or record.report_count <= 0:
        return NumberIntelligence(
            phone_number=phone,
            reported=False,
            report_count=0,
            risk_level="UNKNOWN",
            categories=[],
            last_reported=None,
            message=UNREPORTED_MESSAGE,
        )
    return NumberIntelligence(
        phone_number=record.phone_number,
        reported=True,
        report_count=record.report_count,
        risk_level=record.risk_level,
        categories=_load_categories(record.categories),
        last_reported=record.last_reported,
        message=REPORTED_MESSAGE,
    )


def get_number_intelligence(db: Session, normalized_phone: str) -> NumberIntelligence:
    """Look up an ALREADY-normalized number (used by the analysis pipeline)."""
    return _to_intelligence(normalized_phone, _get_record(db, normalized_phone))


def lookup_number(db: Session, raw_phone: str) -> NumberLookupResponse:
    phone = normalize_phone_number(raw_phone)
    return NumberLookupResponse(**get_number_intelligence(db, phone).model_dump())


def _get_or_create_record(db: Session, phone: str, now) -> PhoneNumber:
    record = _get_record(db, phone)
    if record is not None:
        return record
    record = PhoneNumber(
        phone_number=phone, report_count=0, risk_level="UNKNOWN", categories="[]",
        first_reported=now, last_reported=now,
    )
    db.add(record)
    try:
        db.flush()
    except IntegrityError:  # another request created it at the same moment
        db.rollback()
        record = _get_record(db, phone)
        if record is None:
            raise
    return record


def report_number(db: Session, raw_phone: str, category: str, description: str | None) -> ReportResponse:
    phone = normalize_phone_number(raw_phone)
    now = utcnow()

    record = _get_or_create_record(db, phone, now)

    report = FraudReport(
        phone_number=phone,
        category=category,
        description=(description or "").strip() or None,
        created_at=now,
    )
    db.add(report)

    record.report_count += 1
    record.last_reported = now
    if record.first_reported is None:
        record.first_reported = now
    categories = _load_categories(record.categories)
    if category not in categories:
        categories.append(category)
    record.categories = json.dumps(categories)
    record.risk_level = get_number_risk_level(record.report_count)

    db.commit()
    db.refresh(report)
    db.refresh(record)

    return ReportResponse(
        message="Report submitted. Thank you for helping the community.",
        report_id=report.id,
        number=_to_intelligence(phone, record),
    )
