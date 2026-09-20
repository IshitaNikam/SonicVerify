"""
Seed the database with DEMO / TEST phone numbers.

    python seed_database.py

All numbers below are made-up placeholders (+91 90000 000xx range) for demos and
tests only. Every seeded report description starts with "[DEMO DATA]".
Re-running the script resets these demo numbers (other data is left untouched).
"""
import json
from datetime import timedelta

from sqlalchemy import delete

from app.db.database import SessionLocal, init_db
from app.db.models import FraudReport, PhoneNumber, utcnow
from app.services.risk_service import get_number_risk_level

DEMO_NUMBERS = {
    # Unreported - deliberately NOT inserted; use it to test the "UNKNOWN" path.
    "+919000000001": [],
    # 1 report -> LOW
    "+919000000002": [("OTP fraud", "Caller asked for an OTP 'to verify a delivery'.")],
    # 2 reports -> MEDIUM
    "+919000000003": [
        ("government impersonation", "Claimed to be from a government department."),
        ("financial scam", "Asked for a small 'processing fee' via UPI."),
    ],
    # 3 reports -> MEDIUM
    "+919000000004": [
        ("bank impersonation", "Said the bank account would be blocked."),
        ("OTP fraud", "Requested the OTP received by SMS."),
        ("other", "Repeated calls at odd hours."),
    ],
    # 4 reports -> HIGH
    "+919000000005": [
        ("voice impersonation", "Voice sounded like a relative asking for urgent money."),
        ("financial scam", "Requested an immediate bank transfer."),
        ("voice impersonation", "Same caller, voice sounded robotic at times."),
        ("bank impersonation", "Posed as a bank representative."),
    ],
    # 6 reports -> HIGH
    "+919000000006": [
        ("voice impersonation", "Caller imitated a family member's voice."),
        ("financial scam", "Asked to send money to an unknown account."),
        ("OTP fraud", "Asked for an OTP during the call."),
        ("government impersonation", "Claimed to be a police officer."),
        ("bank impersonation", "Claimed the KYC had expired."),
        ("other", "Threatening tone."),
    ],
}


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        now = utcnow()
        for phone, reports in DEMO_NUMBERS.items():
            # reset this demo number so the script can be re-run safely
            db.execute(delete(FraudReport).where(FraudReport.phone_number == phone))
            db.execute(delete(PhoneNumber).where(PhoneNumber.phone_number == phone))
            if not reports:
                continue

            times = [now - timedelta(days=len(reports) - i, hours=i) for i in range(len(reports))]
            categories: list[str] = []
            for cat, _ in reports:
                if cat not in categories:
                    categories.append(cat)

            db.add(PhoneNumber(
                phone_number=phone,
                report_count=len(reports),
                risk_level=get_number_risk_level(len(reports)),
                categories=json.dumps(categories),
                first_reported=times[0],
                last_reported=times[-1],
            ))
            db.flush()
            for (cat, desc), ts in zip(reports, times):
                db.add(FraudReport(phone_number=phone, category=cat, description=f"[DEMO DATA] {desc}", created_at=ts))
        db.commit()
    finally:
        db.close()

    print("Seeded DEMO data (fictional numbers):")
    for phone, reports in DEMO_NUMBERS.items():
        level = get_number_risk_level(len(reports))
        note = "not stored - use to test unreported numbers" if not reports else f"{len(reports)} report(s)"
        print(f"  {phone}  ->  {level:8s} ({note})")


if __name__ == "__main__":
    seed()
