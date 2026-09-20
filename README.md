# SonicVerify — SIH 2026, PS 26104

# 🎙️ SonicVerify — Audio Deepfake & Voice Scam Detection Backend

SonicVerify is an AI-powered RESTful backend API designed to analyze incoming voice audio calls, detect synthetic deepfakes, evaluate phone number reputation against scam reporting databases, and generate actionable risk assessments in real-time.

---

## 🚀 Key Features

* **Real-time Deepfake Detection:** Integrates machine learning classification models to distinguish authentic human speech from AI-generated audio/voice clones.
* **Reputation Database Evaluation:** Queries reported scam and suspicious phone numbers to assign risk tiers (`LOW`, `MEDIUM`, `HIGH`, `UNKNOWN`).
* **Automated Risk Scoring:** Merges audio verification confidence scores with historical database signals to compute overall threat metrics.
* **Interactive API Documentation:** Built with FastAPI for automated OpenAPI documentation and interactive Swagger UI endpoints.

---

## 🛠️ Tech Stack & Dependencies

* **Framework:** FastAPI / Python 3.10+
* **Database:** SQLite (SQLAlchemy ORM)
* **Testing:** Pytest & HTTPX Async Client
* **Server:** Uvicorn (ASGI)

---

## 📂 Project Architecture

```text
backend/
├── app/
│   ├── api/          # Route handlers & API endpoints
│   ├── db/           # Database models & SQLite session config
│   ├── schemas/      # Pydantic validation models
│   ├── services/     # Audio detection logic & database lookup services
│   ├── config.py     # Global app configuration & settings
│   └── exceptions.py # Custom API error handlers
├── tests/            # Automated unit and API test suite
├── main.py           # FastAPI application entry point
├── seed_database.py  # Script to populate initial demo data
└── requirements.txt  # Python package dependencies

Backend & system-integration layer for **AI-Powered Real-Time Detection and Prevention of Voice Cloning
Impersonation Attacks**. FastAPI + SQLite + SQLAlchemy. Runs locally, no external services.
Designed to be called from a **Streamlit** frontend (also works with any React/JS frontend).

```
Streamlit ──► FastAPI ──► detection_service (ML teammate) ─┐
                     ├──► number database (SQLite)         ├──► risk engine ──► JSON ──► Streamlit
                     └──► call context (financial request, claimed identity) ┘
```
The frontend only talks to FastAPI — never to the database or the model, and never calculates risk itself.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python seed_database.py             # optional: demo numbers
uvicorn main:app --reload
```

Open **http://localhost:8000/docs** (Swagger) and test every endpoint from the browser.
Run tests: `python -m pytest -v` (uses a temporary database — your data is untouched).
Optional settings: copy `.env.example` to `.env`.

## Project structure

```
main.py                     app, CORS, clean-JSON error handlers
seed_database.py            demo data (fictional numbers, re-runnable)
app/config.py               ALL tunables: thresholds, risk weights, context signals, limits, CORS
app/api/                    thin routes: health.py, numbers.py, analysis.py
app/services/
    analysis_service.py     orchestrates: validate → temp file → ML → number lookup → risk → save → delete temp
    detection_service.py    ★ ML integration point (placeholder until the model exists)
    number_service.py       lookup + report + report-count → risk level
    risk_service.py         transparent weighted risk formula
app/db/                     database.py (engine/session), models.py
app/schemas/                Pydantic request/response models
app/utils/validation.py     phone normalization + audio validation
uploads/                    TEMPORARY audio only (emptied after every request)
tests/test_api.py
```

## Database (SQLite, auto-created as `sonicverify.db`)

| Table | Columns |
|---|---|
| `phone_numbers` | id, phone_number (unique), report_count, risk_level, categories (JSON string), first_reported, last_reported |
| `fraud_reports` | id, phone_number (FK → phone_numbers), category, description, created_at |
| `analysis_history` | id, phone_number, synthetic_probability (null if model unavailable), number_risk, context_risk, final_risk_score, risk_level, created_at |

Timestamps are UTC. Delete `sonicverify.db` to reset everything (then re-run `seed_database.py`).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + `model_status` |
| GET | `/api/numbers/{phone_number}` | community-report lookup |
| POST | `/api/numbers/report` | submit a report (JSON body) |
| POST | `/api/analyze` | audio + number + context → full risk result (multipart form) |
| GET | `/api/analysis/history` | latest 20 analyses |

Phone numbers are normalized: `9876543210`, `919876543210`, `+919876543210`, `+91 98765-43210` → `+919876543210`.
Only Indian mobile numbers (10 digits starting 6–9) are accepted in this prototype.

Report categories: `voice impersonation`, `financial scam`, `OTP fraud`, `government impersonation`,
`bank impersonation`, `other`.

### Examples

```bash
curl localhost:8000/api/numbers/+919000000005

curl -X POST localhost:8000/api/numbers/report -H "Content-Type: application/json" \
  -d '{"phone_number":"+919876543210","category":"voice impersonation","description":"Posed as a bank agent"}'

curl -X POST localhost:8000/api/analyze \
  -F "audio=@call.wav" -F "phone_number=+919876543210" \
  -F "financial_request=true" -F "identity_claim=bank representative"
```

`/api/analyze` response (shown with a real model connected):

```json
{
  "success": true,
  "analysis_id": 12,
  "voice_analysis": {
    "synthetic_probability": 0.87, "authentic_probability": 0.13, "confidence": 0.87,
    "model_status": "available",
    "summary": "Possible AI-generated voice detected."
  },
  "number_intelligence": {
    "phone_number": "+919876543210", "reported": true, "report_count": 4, "risk_level": "HIGH",
    "categories": ["voice impersonation", "financial scam"],
    "last_reported": "2026-09-20T08:16:58",
    "message": "Community-reported suspicious number. Community reports are an indicator, not proof of fraud."
  },
  "risk_assessment": {
    "risk_score": 0.94, "risk_level": "HIGH",
    "summary": "High risk: several indicators associated with voice-cloning or impersonation scams were found. This is a risk estimate, not proof.",
    "voice_analysis_used": true,
    "reasons": [
      "Voice analysis indicates characteristics associated with synthetic speech.",
      "This number has previous community reports (4). Community reports are indicators, not proof of fraud.",
      "The conversation involves a financial request."
    ],
    "breakdown": {
      "voice":   {"risk": 0.87, "weight": 0.6,  "contribution": 0.522},
      "number":  {"risk": 1.0,  "weight": 0.25, "contribution": 0.25},
      "context": {"risk": 0.6,  "weight": 0.15, "contribution": 0.09}
    }
  },
  "recommendation": "Do not share OTP, PIN, passwords or banking information. Verify the caller through an independent trusted channel.",
  "disclaimer": "This is an automated risk estimate based on limited signals. It is not proof of fraud or of AI-generated audio."
}
```

**Errors** are always `{"success": false, "error": "..."}` — never a stack trace:
400 invalid phone / empty or corrupted audio · 413 file too large · 415 unsupported format ·
422 missing/invalid field · 500 database/unexpected error (details only in the server log).

### Risk rules (all in `app/config.py`)

* **Number risk level** from report count: 0 → UNKNOWN, 1 → LOW, 2–3 → MEDIUM, 4+ → HIGH.
* **Risk score** = `0.60 × voice + 0.25 × number + 0.15 × context`
  * voice = `synthetic_probability`
  * number = UNKNOWN 0.0 · LOW 0.3 · MEDIUM 0.6 · HIGH 1.0
  * context = financial request (+0.6) + sensitive claimed identity such as bank / police / son / boss (+0.4), max 1.0
* **Final level**: 0.00–0.34 LOW · 0.35–0.64 UNCERTAIN · 0.65–1.00 HIGH.
* These are simple prototype heuristics, not a validated fraud model.

## How ML detection is integrated

Edit **only** `app/services/detection_service.py`:

```python
def detect_voice(audio_path: str) -> dict:
    return {"synthetic_probability": 0.87, "authentic_probability": 0.13,
            "confidence": 0.87, "model_status": "available"}

def get_model_status() -> str:      # shown in /api/health
    return "available"
```

The file documents the full contract and has an example. Routes, risk engine and schemas don't change.
`audio_path` is a temporary file (.wav/.mp3/.m4a) that is deleted right after the call.

Until a real model is connected, the placeholder returns `model_status: "unavailable"` — it **never invents
probabilities**. The same happens (`"error"`) if the model raises an exception or returns invalid values.
In that case `voice_analysis_used` is `false`, the voice weight is dropped, and the remaining weights are
re-scaled, so the score reflects number history + call context only (and the reasons say so).

## Connecting the Streamlit frontend

Streamlit calls the API from its own Python process, so use `requests` (CORS is not involved; `localhost:8501`
is allowed anyway). Base URL: `http://localhost:8000`.

```python
import requests
import streamlit as st

API = "http://localhost:8000"

audio = st.file_uploader("Call recording", type=["wav", "mp3", "m4a"])   # or st.audio_input("Record")
phone = st.text_input("Caller number", "+919876543210")
financial = st.checkbox("Caller asked for money / OTP / bank details")
claim = st.text_input("Caller claims to be (optional)")

if st.button("Analyze") and audio:
    resp = requests.post(
        f"{API}/api/analyze",
        files={"audio": (audio.name, audio.getvalue(), audio.type)},
        data={"phone_number": phone, "financial_request": str(financial).lower(), "identity_claim": claim},
        timeout=120,
    )
    result = resp.json()
    if not result.get("success"):
        st.error(result["error"])                       # clean, user-readable message
    else:
        risk = result["risk_assessment"]
        st.metric("Risk", f'{risk["risk_level"]}  ({risk["risk_score"]:.2f})')
        st.write(risk["summary"])
        for reason in risk["reasons"]:
            st.write("•", reason)
        st.warning(result["recommendation"])
        st.caption(result["disclaimer"])
        if result["voice_analysis"]["model_status"] != "available":
            st.info("Voice model unavailable — result uses number reports and call context only.")

# Report a number
requests.post(f"{API}/api/numbers/report",
              json={"phone_number": phone, "category": "voice impersonation", "description": "..."})
# Lookup
requests.get(f"{API}/api/numbers/{phone}").json()
```

Tips: display `risk_assessment`, `reasons`, `recommendation` and `disclaimer` as returned; always check
`result["success"]` first; `st.audio_input` recordings are WAV and work as-is.

## Privacy

* Audio is written to `uploads/` under a random name, analysed, and **deleted in a `finally` block** — even if
  analysis fails. Leftover temp files (e.g. after a crash) are purged on startup.
* Recordings are never permanently stored, never served through any URL, and never logged.
  `analysis_history` stores only numeric scores and the phone number.
* Report descriptions are stored in `fraud_reports` but are not returned by any endpoint.

## Known limitations

* No authentication or rate limiting — anyone can submit reports, so counts can be inflated. Fine for a demo only.
* `/api/analysis/history` is public and lists phone numbers.
* Only WAV files are fully parsed. MP3/M4A validation checks headers/structure, so a deeply corrupted MP3/M4A
  may pass; the ML module must handle decoding errors (the backend then reports `model_status: "error"`).
* Starlette receives the whole upload before the route runs (spooled to disk); the 25 MB limit is enforced right after.
* Report-count levels and score weights are heuristics. With default weights the voice signal alone reaches at most
  0.60, so a HIGH result normally needs number/context signals too — tune `RISK_WEIGHTS` in `app/config.py`.
* Indian mobile numbers only. Demo numbers (`+91 90000 000xx`) are fictional placeholders but follow the real
  format, so some may coincide with real numbers — don't call them.
