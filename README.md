Here is a crisp, professionally updated `README.md` for your **SonicVerify** repository:

---

# 🎙️ SonicVerify — Audio Deepfake & Voice Scam Detection System

> **SIH 2026 — Problem Statement 26104**
> An end-to-end AI system designed to detect voice cloning, evaluate phone number scam history, and deliver real-time risk assessments.

---

## ⚡ Architecture Overview

```
Streamlit (Frontend) ──► FastAPI (Backend) ──► AASIST ML Model (.pth)
                                             ├──► SQLite (Scam Reports)
                                             └──► Risk & Security Engine

```

---

## 🚀 Key Features

* **Real-Time Deepfake Detection:** Employs the **AASIST** architecture (ASVspoof 2019 pre-trained) to classify audio samples as **SPOOF** (Class 0) or **BONAFIDE** (Class 1).
* **Phone Intelligence:** Queries scam history from an internal SQLite database and assigns risk levels (`LOW`, `MEDIUM`, `HIGH`, `UNKNOWN`).
* **Weighted Threat Scoring:** Combines audio ML confidence, community report counts, and call context (e.g., financial/identity claims) to produce an overall risk score ($0.0 - 1.0$).
* **Automated Security Actions:** Maps final risk levels to actions: **ALLOW** (`LOW`), **WARNING** (`UNCERTAIN`), or **VERIFY** (`HIGH`).
* **Privacy-First Design:** Audio files are stored temporarily in `uploads/` during inference and deleted immediately in a `finally` block.

---

## 🛠️ Project Structure

```text
C:\voice_deepfake\
├── main.py                   # FastAPI entry point & CORS configuration
├── inference.py              # ML inference pipeline wrapper
├── AASIST.py                 # AASIST model neural network architecture
├── AASIST.pth                # Pre-trained model weights
├── seed_database.py          # Script to populate initial demo numbers
├── requirements-ml.txt       # Combined ML & backend Python dependencies
├── app/
│   ├── api/                  # API routes (health, numbers, analysis)
│   ├── db/                   # Database models & SQLite configuration
│   ├── schemas/              # Pydantic request/response validation schemas
│   └── services/             # Core logic (ML detection, risk scoring, security)
└── uploads/                  # Temporary processing directory (auto-cleared)

```

---

## 🎯 Model & Audio Contract

* **Input Specifications:** Mono, 16 kHz WAV/MP3/M4A (~4-second audio chunks / 64,600 samples).
* **ML Output:** Returns `synthetic_probability`, `authentic_probability`, `confidence`, and `model_status`.
* **Graceful Degradation:** If the model is offline or encounters an unsupported file format, the system recalculates risk using only community reports and call context without failing the API request.

---

## 📦 Local Setup & Execution

### 1. Installation

```bash
git clone https://github.com/IshitaNikam/SonicVerify.git
cd SonicVerify
pip install -r requirements-ml.txt

```

### 2. Seed Database (Optional)

```bash
python seed_database.py

```

### 3. Run the Backend

```bash
python -m uvicorn main:app --reload

```

* **Interactive API Docs (Swagger):** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

---

## 📡 Core API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service health status & ML model availability |
| `GET` | `/api/numbers/{phone}` | Look up community report history for a phone number |
| `POST` | `/api/numbers/report` | Submit a scam/impersonation report for a number |
| `POST` | `/api/analyze` | Submit multipart form data (`audio`, `phone_number`, `financial_request`, `identity_claim`) for deepfake detection & full risk analysis |
| `GET` | `/api/analysis/history` | Retrieve recent analysis logs |

---

## 🌐 Live Deployment

* **Live API Backend (Render):** `[https://sonicverify.onrender.com](https://sonicverify.onrender.com)`
* **API Documentation:** `[https://sonicverify.onrender.com/docs](https://sonicverify.onrender.com/docs)`
