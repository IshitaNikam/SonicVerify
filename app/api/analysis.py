from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.analysis import AnalysisResponse, HistoryResponse
from app.services import analysis_service

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze a call recording")
def analyze(
    audio: UploadFile = File(..., description="WAV, MP3 or M4A file, max 25 MB"),
    phone_number: str = Form(..., description="Caller's number, e.g. +919876543210"),
    financial_request: bool = Form(False, description="Did the caller ask for money / OTP / bank details?"),
    identity_claim: Optional[str] = Form(None, description="Who the caller claims to be, e.g. 'bank representative'"),
    db: Session = Depends(get_db),
):
    """
    Full pipeline: validate audio -> ML voice detection -> number lookup -> risk engine.
    The uploaded audio is kept only temporarily and deleted right after analysis.
    """
    return analysis_service.run_analysis(db, audio, phone_number, financial_request, identity_claim)


@router.get("/analysis/history", response_model=HistoryResponse, summary="Latest 20 analyses")
def analysis_history(db: Session = Depends(get_db)):
    return analysis_service.get_history(db)
