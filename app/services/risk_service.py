"""
Risk engine: a simple, transparent weighted formula (see app/config.py).

    final_score = voice_risk*0.60 + number_risk*0.25 + context_risk*0.15

  voice_risk   = synthetic_probability from the ML model
  number_risk  = value of the number's community-report risk level
  context_risk = financial request + sensitive identity claim signals

If the ML model is unavailable, the voice weight is dropped and the other two
weights are re-scaled to sum to 1, so the score is based on number + context only
(and the response says so). Nothing is faked.
"""
import re
from dataclasses import dataclass
from typing import Optional

from app import config
from app.schemas.analysis import RiskAssessment, RiskBreakdown, ScoreComponent

DISCLAIMER = (
    "This is an automated risk estimate based on limited signals. It is not proof of fraud "
    "or of AI-generated audio."
)

_IDENTITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in config.SENSITIVE_IDENTITY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Number risk
# ---------------------------------------------------------------------------
def get_number_risk_level(report_count: int) -> str:
    """0 reports -> UNKNOWN; otherwise the first matching threshold in config."""
    for min_reports, level in config.NUMBER_RISK_THRESHOLDS:
        if report_count >= min_reports:
            return level
    return "UNKNOWN"


def number_risk_value(risk_level: str) -> float:
    return config.NUMBER_RISK_VALUES.get(risk_level, 0.0)


# ---------------------------------------------------------------------------
# Context risk
# ---------------------------------------------------------------------------
def is_sensitive_identity(identity_claim: Optional[str]) -> bool:
    return bool(identity_claim and _IDENTITY_PATTERN.search(identity_claim))


def calculate_context_risk(financial_request: bool, identity_claim: Optional[str]) -> float:
    risk = 0.0
    if financial_request:
        risk += config.CONTEXT_SIGNALS["financial_request"]
    if is_sensitive_identity(identity_claim):
        risk += config.CONTEXT_SIGNALS["sensitive_identity_claim"]
    return min(risk, 1.0)


# ---------------------------------------------------------------------------
# Final level
# ---------------------------------------------------------------------------
def score_to_level(score: float) -> str:
    if score >= config.RISK_LEVEL_HIGH_MIN:
        return "HIGH"
    if score >= config.RISK_LEVEL_UNCERTAIN_MIN:
        return "UNCERTAIN"
    return "LOW"


def voice_summary(synthetic_probability: Optional[float]) -> str:
    """Careful wording - a probability, never a certainty."""
    if synthetic_probability is None:
        return "Voice analysis is unavailable. The result is based on number reports and call context only."
    if synthetic_probability >= config.RISK_LEVEL_HIGH_MIN:
        return "Possible AI-generated voice detected."
    if synthetic_probability >= config.RISK_LEVEL_UNCERTAIN_MIN:
        return "Voice analysis is inconclusive."
    return "No strong indicators of synthetic speech were found."


_LEVEL_SUMMARY = {
    "HIGH": "High risk: several indicators associated with voice-cloning or impersonation scams were found. "
            "This is a risk estimate, not proof.",
    "UNCERTAIN": "Uncertain: some risk indicators were found. Please verify the caller independently.",
    "LOW": "Low risk: no strong indicators were found. This is an estimate, not a guarantee.",
}

# Used instead of _LEVEL_SUMMARY when the voice model did not produce a result,
# so the wording never implies that a cloned voice was detected.
_LEVEL_SUMMARY_NO_VOICE = {
    "HIGH": "High risk: the number's report history and the call context match common impersonation-scam patterns. "
            "Voice analysis was not available. This is a risk estimate, not proof.",
    "UNCERTAIN": "Uncertain: some risk indicators were found (voice analysis was not available). "
                 "Please verify the caller independently.",
    "LOW": "Low risk based on number history and call context only (voice analysis was not available). "
           "This is an estimate, not a guarantee.",
}

_RECOMMENDATION = {
    "HIGH": "Do not share OTP, PIN, passwords or banking information. "
            "Verify the caller through an independent trusted channel.",
    "UNCERTAIN": "Treat this call with caution. Do not share OTP, PIN, passwords or banking information "
                 "until you have verified the caller through an independent trusted channel "
                 "(for example, call back on an official number).",
    "LOW": "No strong risk indicators were found, but this is not a guarantee. Never share OTP, PIN or "
           "passwords on a call, and verify any unexpected request through a trusted channel.",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass
class RiskResult:
    assessment: RiskAssessment
    recommendation: str
    number_risk: float
    context_risk: float


def assess_risk(
    synthetic_probability: Optional[float],
    number_risk_level: str,
    report_count: int,
    financial_request: bool,
    identity_claim: Optional[str],
) -> RiskResult:
    voice_used = synthetic_probability is not None

    voice_risk = synthetic_probability if voice_used else None
    number_risk = number_risk_value(number_risk_level)
    context_risk = calculate_context_risk(financial_request, identity_claim)

    # Weights (re-scale if the voice model was unavailable)
    weights = dict(config.RISK_WEIGHTS)
    if not voice_used:
        remaining = weights["number"] + weights["context"]
        weights = {
            "voice": 0.0,
            "number": weights["number"] / remaining,
            "context": weights["context"] / remaining,
        }

    voice_contrib = (voice_risk or 0.0) * weights["voice"]
    number_contrib = number_risk * weights["number"]
    context_contrib = context_risk * weights["context"]

    score = round(min(max(voice_contrib + number_contrib + context_contrib, 0.0), 1.0), 2)
    level = score_to_level(score)

    # Human-readable reasons
    reasons: list[str] = []
    if not voice_used:
        reasons.append("Voice analysis was unavailable, so this assessment uses number reports and call context only.")
    elif synthetic_probability >= config.RISK_LEVEL_HIGH_MIN:
        reasons.append("Voice analysis indicates characteristics associated with synthetic speech.")
    elif synthetic_probability >= config.RISK_LEVEL_UNCERTAIN_MIN:
        reasons.append("Voice analysis was inconclusive; some characteristics may be associated with synthetic speech.")
    else:
        reasons.append("Voice analysis did not find strong indicators of synthetic speech.")

    if report_count > 0:
        reasons.append(
            f"This number has previous community reports ({report_count}). "
            "Community reports are indicators, not proof of fraud."
        )
    else:
        reasons.append("No community reports were found for this number (this does not guarantee it is safe).")

    if financial_request:
        reasons.append("The conversation involves a financial request.")
    if is_sensitive_identity(identity_claim):
        reasons.append("The caller claims an identity that is commonly used in impersonation scams.")

    assessment = RiskAssessment(
        risk_score=score,
        risk_level=level,
        summary=(_LEVEL_SUMMARY if voice_used else _LEVEL_SUMMARY_NO_VOICE)[level],
        voice_analysis_used=voice_used,
        reasons=reasons,
        breakdown=RiskBreakdown(
            voice=ScoreComponent(
                risk=round(voice_risk, 4) if voice_used else None,
                weight=round(weights["voice"], 4),
                contribution=round(voice_contrib, 4),
            ),
            number=ScoreComponent(
                risk=round(number_risk, 4), weight=round(weights["number"], 4), contribution=round(number_contrib, 4)
            ),
            context=ScoreComponent(
                risk=round(context_risk, 4), weight=round(weights["context"], 4), contribution=round(context_contrib, 4)
            ),
        ),
    )
    return RiskResult(
        assessment=assessment,
        recommendation=_RECOMMENDATION[level],
        number_risk=number_risk,
        context_risk=context_risk,
    )
