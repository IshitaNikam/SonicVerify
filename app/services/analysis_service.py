"""
Orchestrates the whole analysis pipeline. The API route stays thin.

    validate -> temp file -> ML detection -> number lookup -> risk engine
             -> save history -> delete temp file -> response
"""
import logging
import uuid
from typing import Optional

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.db.models import AnalysisHistory
from app.exceptions import FileTooLargeError
from app.schemas.analysis import AnalysisResponse, HistoryItem, HistoryResponse, VoiceAnalysis
from app.services import number_service, risk_service,security_service
from app.services.detection_service import detect_voice
from app.utils.validation import normalize_phone_number, validate_audio_upload

logger = logging.getLogger("sonicverify")

MAX_IDENTITY_LENGTH = 200


def _unavailable(status: str) -> dict:
    return {"synthetic_probability": None, "authentic_probability": None, "confidence": None, "model_status": status}


def _is_prob(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _run_detection(audio_path: str) -> dict:
    """Call the ML module and make sure its output is sane. Never raises."""
    try:
        raw = detect_voice(audio_path)
    except Exception:  # noqa: BLE001 - the ML module must never crash the API
        logger.exception("Voice detection module raised an error")
        return _unavailable("error")

    if not isinstance(raw, dict) or raw.get("model_status") != "available":
        status = raw.get("model_status") if isinstance(raw, dict) else None
        return _unavailable(status if status in ("unavailable", "error") else "unavailable")

    synthetic = raw.get("synthetic_probability")
    if not _is_prob(synthetic):
        logger.error("Detection module returned an invalid synthetic_probability: %r", synthetic)
        return _unavailable("error")
    synthetic = float(synthetic)

    authentic = raw.get("authentic_probability")
    authentic = float(authentic) if _is_prob(authentic) else round(1.0 - synthetic, 4)
    confidence = raw.get("confidence")
    confidence = float(confidence) if _is_prob(confidence) else max(synthetic, authentic)

    return {
        "synthetic_probability": synthetic,
        "authentic_probability": authentic,
        "confidence": confidence,
        "model_status": "available",
    }


def _read_upload(upload: UploadFile) -> bytes:
    """Read at most MAX+1 bytes so a huge upload is never loaded fully into memory."""
    data = upload.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise FileTooLargeError(f"File too large. Maximum allowed size is {config.MAX_UPLOAD_MB} MB.")
    return data


def run_analysis(
    db: Session,
    upload: UploadFile,
    phone_number: str,
    financial_request: bool = False,
    identity_claim: Optional[str] = None,
) -> AnalysisResponse:
    # 1. Validate inputs (cheap checks first)
    phone = normalize_phone_number(phone_number)
    identity_claim = (identity_claim or "").strip()[:MAX_IDENTITY_LENGTH] or None
    data = _read_upload(upload)
    extension = validate_audio_upload(upload.filename, data)

    # 2. Temporary storage - always removed, even if something fails below
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = config.UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}{extension}"
    try:
        temp_path.write_bytes(data)

        # 3. ML detection
        detection = _run_detection(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    # 4. Number lookup
    number_info = number_service.get_number_intelligence(db, phone)

    # 5. Risk engine
    risk = risk_service.assess_risk(
        synthetic_probability=detection["synthetic_probability"],
        number_risk_level=number_info.risk_level,
        report_count=number_info.report_count,
        financial_request=financial_request,
        identity_claim=identity_claim,
    )
    security_result = security_service.security_check(
    risk.assessment.risk_level
)

    # 6. Save history (numbers only - no audio)
    history = AnalysisHistory(
        phone_number=phone,
        synthetic_probability=detection["synthetic_probability"],
        number_risk=risk.number_risk,
        context_risk=risk.context_risk,
        final_risk_score=risk.assessment.risk_score,
        risk_level=risk.assessment.risk_level,
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    # 7. Final response
    return AnalysisResponse(
        analysis_id=history.id,
        voice_analysis=VoiceAnalysis(
            **detection, summary=risk_service.voice_summary(detection["synthetic_probability"])
        ),
        number_intelligence=number_info,
        risk_assessment=risk.assessment,
        recommendation=risk.recommendation,
        disclaimer=risk_service.DISCLAIMER,
    )


def get_history(db: Session) -> HistoryResponse:
    rows = db.execute(
        select(AnalysisHistory)
        .order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc())
        .limit(config.HISTORY_LIMIT)
    ).scalars().all()
    items = [HistoryItem.model_validate(r) for r in rows]
    return HistoryResponse(count=len(items), items=items)


def cleanup_stale_uploads() -> int:
    """Delete leftover temp audio (e.g. after a crash). Called on startup."""
    removed = 0
    if config.UPLOAD_DIR.exists():
        for path in config.UPLOAD_DIR.glob("tmp_*"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
