"""Database tables."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite stores datetimes without timezone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PhoneNumber(Base):
    __tablename__ = "phone_numbers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    phone_number: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    report_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), default="UNKNOWN", nullable=False)
    categories: Mapped[str] = mapped_column(Text, default="[]", nullable=False)  # JSON list as string
    first_reported: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_reported: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    reports: Mapped[list["FraudReport"]] = relationship(back_populates="number", cascade="all, delete-orphan")


class FraudReport(Base):
    __tablename__ = "fraud_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    phone_number: Mapped[str] = mapped_column(
        String(20), ForeignKey("phone_numbers.phone_number"), index=True, nullable=False
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    number: Mapped[PhoneNumber] = relationship(back_populates="reports")


class AnalysisHistory(Base):
    """One row per analysis. Audio is NEVER stored - only the numeric results."""

    __tablename__ = "analysis_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    phone_number: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    synthetic_probability: Mapped[float | None] = mapped_column(Float, nullable=True)  # None if model unavailable
    number_risk: Mapped[float] = mapped_column(Float, nullable=False)
    context_risk: Mapped[float] = mapped_column(Float, nullable=False)
    final_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
