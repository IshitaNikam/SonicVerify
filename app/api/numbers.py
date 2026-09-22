from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from pydantic import BaseModel
from typing import Optional

class ReportRequest(BaseModel):
    phone_number: str
    category: Optional[str] = "spam"
    description: Optional[str] = ""

class ReportResponse(BaseModel):
    success: bool
    message: str

class NumberLookupResponse(BaseModel):
    phone_number: str
    risk_level: str
    report_count: int
from app.services import number_service

router = APIRouter(prefix="/api/numbers", tags=["Numbers"])


@router.post("/report", response_model=ReportResponse, status_code=201, summary="Report a suspicious number")
def report_number(payload: ReportRequest, db: Session = Depends(get_db)):
    """Adds a community report, increments the report count and recalculates the number's risk level."""
    return number_service.report_number(db, payload.phone_number, payload.category, payload.description)


@router.get("/{phone_number}", response_model=NumberLookupResponse, summary="Look up a phone number")
def lookup_number(phone_number: str, db: Session = Depends(get_db)):
    """Accepts 9876543210, 919876543210 or +919876543210 - all resolve to the same number."""
    return number_service.lookup_number(db, phone_number)
