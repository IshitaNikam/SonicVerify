from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.numbers import NumberIntelligence


class VoiceAnalysis(BaseModel):
    synthetic_probability: Optional[float] = None
    authentic_probability: Optional[float] = None
    confidence: Optional[float] = None
    model_status: str  # "available" | "unavailable" | "error"
    summary: str


class ScoreComponent(BaseModel):
    risk: Optional[float] = None   # 0-1 input signal (None = not available)
    weight: float                  # weight actually used
    contribution: float            # risk * weight


class RiskBreakdown(BaseModel):
    voice: ScoreComponent
    number: ScoreComponent
    context: ScoreComponent


class RiskAssessment(BaseModel):
    risk_score: float
    risk_level: str  # LOW | UNCERTAIN | HIGH
    summary: str
    voice_analysis_used: bool
    reasons: list[str]
    breakdown: RiskBreakdown


class AnalysisResponse(BaseModel):
    success: bool = True
    analysis_id: int
    voice_analysis: VoiceAnalysis
    number_intelligence: NumberIntelligence
    risk_assessment: RiskAssessment
    security_action: str
    verification_required: bool
    recommendation: str
    disclaimer: str
    


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    phone_number: str
    synthetic_probability: Optional[float] = None
    number_risk: float
    context_risk: float
    final_risk_score: float
    risk_level: str
    created_at: datetime


class HistoryResponse(BaseModel):
    success: bool = True
    count: int
    items: list[HistoryItem]
