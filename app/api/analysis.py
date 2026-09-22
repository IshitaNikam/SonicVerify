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
    financial_request: bool = Form(False),
    identity_claim: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    return analysis_service.run_analysis(
        db, audio, financial_request, identity_claim
    )




@router.get("/analysis/history", response_model=HistoryResponse, summary="Latest 20 analyses")
def analysis_history(db: Session = Depends(get_db)):
    return analysis_service.get_history(db)
