"""
Basic API tests.  Run from the backend/ folder:   python -m pytest -v
Uses a throw-away SQLite file and upload folder, so your real data is never touched.
"""
import io
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

# --- point the app at temporary storage BEFORE importing it -----------------
_TMP = tempfile.mkdtemp(prefix="voiceguard_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(Path(_TMP, "uploads"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.db.database import Base, engine  # noqa: E402
from app.exceptions import InvalidPhoneNumberError  # noqa: E402
from app.services.risk_service import get_number_risk_level  # noqa: E402
from app.utils.validation import normalize_phone_number, validate_audio_upload  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)
PHONE = "+919876543210"


@pytest.fixture(autouse=True)
def fresh_db():
    from app.db import models  # noqa: F401

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def make_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(rate * seconds))
        ))
    return buf.getvalue()


def analyze(phone=PHONE, filename="call.wav", content=None, content_type="audio/wav", **fields):
    content = make_wav() if content is None else content
    data = {"phone_number": phone, **{k: str(v).lower() if isinstance(v, bool) else v for k, v in fields.items()}}
    return client.post("/api/analyze", files={"audio": (filename, content, content_type)}, data=data)


def report(phone=PHONE, category="voice impersonation", description="Test report"):
    return client.post("/api/numbers/report", json={"phone_number": phone, "category": category, "description": description})


# 1. Health ------------------------------------------------------------------
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["service"] == "VoiceGuard Backend"


# 2. Normalization -----------------------------------------------------------
@pytest.mark.parametrize("raw", ["9876543210", "919876543210", "+919876543210", "+91 98765-43210", "09876543210", "00919876543210"])
def test_normalization_accepts_variants(raw):
    assert normalize_phone_number(raw) == PHONE


@pytest.mark.parametrize("raw", ["", "   ", "12345", "abcdefghij", "5876543210", "9999999999", "+14155552671", "98765432101234"])
def test_normalization_rejects_invalid(raw):
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number(raw)


# 3. Unknown number lookup ---------------------------------------------------
def test_unknown_number_lookup():
    r = client.get("/api/numbers/9876543210")
    assert r.status_code == 200
    body = r.json()
    assert body["phone_number"] == PHONE
    assert body["reported"] is False
    assert body["report_count"] == 0
    assert body["risk_level"] == "UNKNOWN"
    assert body["categories"] == []


# 4 + 5. Report creation and count increment ---------------------------------
def test_report_creation_and_increment():
    r = report(phone="9876543210")
    assert r.status_code == 201
    assert r.json()["number"]["report_count"] == 1

    r = report(phone="+91 98765 43210", category="financial scam")
    assert r.json()["number"]["report_count"] == 2

    lookup = client.get(f"/api/numbers/{PHONE}").json()
    assert lookup["reported"] is True
    assert lookup["report_count"] == 2
    assert set(lookup["categories"]) == {"voice impersonation", "financial scam"}
    assert "Community-reported suspicious number" in lookup["message"]


def test_report_rejects_bad_category():
    r = report(category="not a category")
    assert r.status_code == 422
    assert r.json()["success"] is False


# 6. Number risk calculation -------------------------------------------------
@pytest.mark.parametrize("count,level", [(0, "UNKNOWN"), (1, "LOW"), (2, "MEDIUM"), (3, "MEDIUM"), (4, "HIGH"), (10, "HIGH")])
def test_number_risk_levels(count, level):
    assert get_number_risk_level(count) == level


def test_risk_level_updates_via_api():
    levels = [report().json()["number"]["risk_level"] for _ in range(4)]
    assert levels == ["LOW", "MEDIUM", "MEDIUM", "HIGH"]


# 7. Invalid phone number ----------------------------------------------------
def test_invalid_phone_number_everywhere():
    r = client.get("/api/numbers/12345")
    assert r.status_code == 400 and r.json()["success"] is False

    r = report(phone="12345")
    assert r.status_code == 400 and r.json()["success"] is False

    r = analyze(phone="12345")
    assert r.status_code == 400 and r.json()["success"] is False


def test_missing_phone_number_on_analyze():
    r = client.post("/api/analyze", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False
    assert "phone_number" in body["error"]


# 8. Invalid audio -----------------------------------------------------------
def test_invalid_audio_uploads():
    r = analyze(filename="call.txt", content=b"hello", content_type="text/plain")
    assert r.status_code == 415 and r.json()["success"] is False
    assert "Unsupported audio format" in r.json()["error"]

    r = analyze(content=b"")
    assert r.status_code == 400 and "empty" in r.json()["error"].lower()

    r = analyze(content=b"this is not a wav file at all")
    assert r.status_code == 400 and "corrupted" in r.json()["error"].lower()

    r = analyze(content=make_wav()[:60])  # truncated WAV
    assert r.status_code == 400 and r.json()["success"] is False


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    r = analyze(content=make_wav())
    assert r.status_code == 413 and r.json()["success"] is False


def test_mp3_and_m4a_validation():
    fake_mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" + b"\x00" * 100
    assert validate_audio_upload("x.mp3", fake_mp3) == ".mp3"

    def box(kind, payload=b""):
        return struct.pack(">I", 8 + len(payload)) + kind + payload

    fake_m4a = box(b"ftyp", b"M4A \x00\x00\x00\x00") + box(b"moov", b"\x00" * 8) + box(b"mdat", b"\x00" * 16)
    assert validate_audio_upload("x.m4a", fake_m4a) == ".m4a"

    from app.exceptions import CorruptedAudioError
    with pytest.raises(CorruptedAudioError):
        validate_audio_upload("x.mp3", b"just some text")
    with pytest.raises(CorruptedAudioError):
        validate_audio_upload("x.m4a", b"just some text, not m4a")


# 9. Complete analysis -------------------------------------------------------
def test_full_analysis_with_placeholder_model():
    """Placeholder detector -> model unavailable; must NOT invent probabilities."""
    r = analyze(financial_request=True, identity_claim="bank representative")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["voice_analysis"]["model_status"] == "unavailable"
    assert body["voice_analysis"]["synthetic_probability"] is None
    assert body["risk_assessment"]["voice_analysis_used"] is False
    assert body["risk_assessment"]["risk_level"] in {"LOW", "UNCERTAIN", "HIGH"}
    assert body["number_intelligence"]["phone_number"] == PHONE
    assert body["analysis_id"] >= 1
    assert body["recommendation"]
    # temp audio must be gone
    assert list(config.UPLOAD_DIR.glob("tmp_*")) == []


def test_full_analysis_with_available_model_and_report_flow(monkeypatch):
    from app.services import analysis_service

    monkeypatch.setattr(
        analysis_service, "detect_voice",
        lambda path: {"synthetic_probability": 0.9, "authentic_probability": 0.1, "confidence": 0.9, "model_status": "available"},
    )

    # Before any report: 0.9*0.6 + 0*0.25 + 0.15*1.0 = 0.69 -> HIGH (financial + bank claim)
    first = analyze(financial_request=True, identity_claim="Bank Representative").json()
    assert first["number_intelligence"]["reported"] is False
    assert first["risk_assessment"]["voice_analysis_used"] is True
    assert first["risk_assessment"]["risk_score"] == 0.69
    assert first["risk_assessment"]["risk_level"] == "HIGH"

    # Voice only, no context, no reports: 0.9*0.6 = 0.54 -> UNCERTAIN
    plain = analyze().json()
    assert plain["risk_assessment"]["risk_score"] == 0.54
    assert plain["risk_assessment"]["risk_level"] == "UNCERTAIN"

    # Report the number 4 times -> future analyses see the higher number risk
    for _ in range(4):
        report()
    after = analyze().json()
    assert after["number_intelligence"]["report_count"] == 4
    assert after["number_intelligence"]["risk_level"] == "HIGH"
    assert after["risk_assessment"]["risk_score"] == 0.79  # 0.54 + 1.0*0.25
    assert after["risk_assessment"]["risk_score"] > plain["risk_assessment"]["risk_score"]


def test_model_error_is_handled(monkeypatch):
    from app.services import analysis_service

    def boom(path):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(analysis_service, "detect_voice", boom)
    r = analyze()
    assert r.status_code == 200
    assert r.json()["voice_analysis"]["model_status"] == "error"


# 10. History ----------------------------------------------------------------
def test_analysis_history():
    assert client.get("/api/analysis/history").json()["count"] == 0
    for _ in range(3):
        assert analyze().status_code == 200
    body = client.get("/api/analysis/history").json()
    assert body["success"] is True
    assert body["count"] == 3
    assert body["items"][0]["id"] > body["items"][-1]["id"]  # newest first
    assert body["items"][0]["phone_number"] == PHONE


def test_history_limited_to_20():
    for _ in range(22):
        analyze()
    assert client.get("/api/analysis/history").json()["count"] == 20
