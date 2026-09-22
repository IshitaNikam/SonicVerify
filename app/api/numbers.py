from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/numbers", tags=["Numbers"])

class ReportRequest(BaseModel):
    phone_number: str
    category: Optional[str] = "spam"
    description: Optional[str] = ""

class ReportResponse(BaseModel):
    success: bool = True
    message: str = "Report submitted successfully"

class NumberLookupResponse(BaseModel):
    phone_number: str
    risk_level: str = "LOW"
    report_count: int = 0

@router.post("/report", response_model=ReportResponse, status_code=201)
def report_number(payload: ReportRequest):
    return ReportResponse()

@router.get("/{phone_number}", response_model=NumberLookupResponse)
def lookup_number(phone_number: str):
    return NumberLookupResponse(phone_number=phone_number)
