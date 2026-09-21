from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app import config


class NumberIntelligence(BaseModel):
    """Community-report information about a phone number (used by lookup AND analysis)."""

    phone_number: str
    reported: bool
    report_count: int
    risk_level: str = Field(description="UNKNOWN | LOW | MEDIUM | HIGH")
    categories: list[str]
    last_reported: Optional[datetime] = None
    message: str


class NumberLookupResponse(NumberIntelligence):
    success: bool = True


class ReportRequest(BaseModel):
    phone_number: str = Field(examples=["+919876543210"])
    category: str = Field(examples=["voice impersonation"])
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        examples=["Caller appeared to impersonate a bank representative."],
    )

    @field_validator("category")
    @classmethod
    def _check_category(cls, value: str) -> str:
        cleaned = value.strip()
        for valid in config.VALID_CATEGORIES:
            if cleaned.lower() == valid.lower():
                return valid
        raise ValueError(f"Invalid category. Allowed categories: {', '.join(config.VALID_CATEGORIES)}.")


class ReportResponse(BaseModel):
    success: bool = True
    message: str
    report_id: int
    number: NumberIntelligence
