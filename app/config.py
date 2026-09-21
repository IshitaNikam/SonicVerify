"""
Central configuration for the sonicverify backend.

Everything you might want to tune during the hackathon lives in this file:
  * report-count -> number risk thresholds
  * risk-score weights
  * context signals
  * final risk-level cut-offs
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_dir(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
APP_NAME = "sonicverify Backend"
APP_VERSION = "1.0.0"

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(BASE_DIR / 'sonicverify.db').as_posix()}"

CORS_ORIGINS = _env_list(
    "CORS_ORIGINS",
    [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
    ],
)

# --------------------------------------------------------------------------
# Audio upload rules
# --------------------------------------------------------------------------
UPLOAD_DIR = _resolve_dir(os.getenv("UPLOAD_DIR"), BASE_DIR / "uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}

# --------------------------------------------------------------------------
# Phone-number reports
# --------------------------------------------------------------------------
VALID_CATEGORIES = [
    "voice impersonation",
    "financial scam",
    "OTP fraud",
    "government impersonation",
    "bank impersonation",
    "other",
]

# (minimum report count, risk level) - checked from the top; first match wins.
# 0 reports -> UNKNOWN. These are prototype heuristics, NOT a validated fraud model.
NUMBER_RISK_THRESHOLDS = [
    (4, "HIGH"),
    (2, "MEDIUM"),
    (1, "LOW"),
]

# Numeric value (0-1) each number risk level contributes to the final risk formula.
NUMBER_RISK_VALUES = {
    "UNKNOWN": 0.0,
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 1.0,
}

# --------------------------------------------------------------------------
# Risk engine
# --------------------------------------------------------------------------
# final_score = voice*0.60 + number*0.25 + context*0.15   (each input is 0-1)
RISK_WEIGHTS = {
    "voice": 0.60,
    "number": 0.25,
    "context": 0.15,
}

# Context risk (0-1) = sum of the signals that apply, capped at 1.0
CONTEXT_SIGNALS = {
    "financial_request": 0.6,
    "sensitive_identity_claim": 0.4,
}

# Words that make a claimed identity "sensitive" (commonly impersonated in scams).
# Matched as whole words, case-insensitive.
SENSITIVE_IDENTITY_KEYWORDS = [
    "bank", "manager", "government", "police", "cbi", "rbi", "income tax",
    "customs", "court", "officer", "official", "kyc", "insurance", "telecom",
    "trai", "relative", "family", "son", "daughter", "brother", "sister",
    "mother", "father", "boss", "ceo", "employer", "colleague",
]

# Final score -> level.   score < UNCERTAIN_MIN -> LOW,
# UNCERTAIN_MIN <= score < HIGH_MIN -> UNCERTAIN,  score >= HIGH_MIN -> HIGH
RISK_LEVEL_UNCERTAIN_MIN = 0.35
RISK_LEVEL_HIGH_MIN = 0.65

HISTORY_LIMIT = 20
