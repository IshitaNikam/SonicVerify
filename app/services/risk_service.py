"""
SonicVerify Risk Engine

The risk score is based on:

    Voice risk   = synthetic probability from the ML model
    Context risk = financial request + sensitive identity claim

Phone-number analysis is NOT used in the voice-analysis pipeline.

Risk weights:

    Voice   = 80%
    Context = 20%
    Number  = 0%

The result is an automated risk estimate, not proof of fraud
or proof that an audio recording is AI-generated.
"""

import re
from dataclasses import dataclass
from typing import Optional

from app import config
from app.schemas.analysis import (
    RiskAssessment,
    RiskBreakdown,
    ScoreComponent,
)


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This is an automated risk estimate based on limited signals. "
    "It is not proof of fraud or of AI-generated audio."
)


# ---------------------------------------------------------------------------
# Sensitive identity detection
# ---------------------------------------------------------------------------

_IDENTITY_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(k)
        for k in config.SENSITIVE_IDENTITY_KEYWORDS
    )
    + r")\b",
    re.IGNORECASE,
)


def is_sensitive_identity(
    identity_claim: Optional[str],
) -> bool:
    """
    Check whether the caller claims a sensitive identity.

    Example:
        "bank representative"
        "police officer"
        "government official"
    """

    return bool(
        identity_claim
        and _IDENTITY_PATTERN.search(identity_claim)
    )


# ---------------------------------------------------------------------------
# Context risk
# ---------------------------------------------------------------------------

def calculate_context_risk(
    financial_request: bool,
    identity_claim: Optional[str],
) -> float:
    """
    Calculate risk based on the context of the conversation.

    Signals:
        - Financial / OTP / banking request
        - Sensitive identity claim
    """

    risk = 0.0

    if financial_request:
        risk += config.CONTEXT_SIGNALS["financial_request"]

    if is_sensitive_identity(identity_claim):
        risk += config.CONTEXT_SIGNALS[
            "sensitive_identity_claim"
        ]

    return min(risk, 1.0)


# ---------------------------------------------------------------------------
# Final risk level
# ---------------------------------------------------------------------------

def score_to_level(score: float) -> str:
    """
    Convert a numerical risk score into a risk level.
    """

    if score >= config.RISK_LEVEL_HIGH_MIN:
        return "HIGH"

    if score >= config.RISK_LEVEL_UNCERTAIN_MIN:
        return "UNCERTAIN"

    return "LOW"


# ---------------------------------------------------------------------------
# Voice analysis summary
# ---------------------------------------------------------------------------

def voice_summary(
    synthetic_probability: Optional[float],
) -> str:
    """
    Generate a human-readable summary of the ML result.

    The wording intentionally avoids claiming certainty.
    """

    if synthetic_probability is None:
        return (
            "Voice analysis is unavailable. "
            "The result is based on call context only."
        )

    if (
        synthetic_probability
        >= config.RISK_LEVEL_HIGH_MIN
    ):
        return "Possible AI-generated voice detected."

    if (
        synthetic_probability
        >= config.RISK_LEVEL_UNCERTAIN_MIN
    ):
        return "Voice analysis is inconclusive."

    return (
        "No strong indicators of synthetic speech "
        "were found."
    )


# ---------------------------------------------------------------------------
# Risk summaries
# ---------------------------------------------------------------------------

_LEVEL_SUMMARY = {
    "HIGH": (
        "High risk: the voice and/or call context contains "
        "indicators associated with voice-cloning or "
        "impersonation scams. This is a risk estimate, not proof."
    ),

    "UNCERTAIN": (
        "Uncertain: some risk indicators were found. "
        "Please verify the caller independently."
    ),

    "LOW": (
        "Low risk: no strong indicators were found. "
        "This is an estimate, not a guarantee."
    ),
}


_LEVEL_SUMMARY_NO_VOICE = {
    "HIGH": (
        "High risk: the call context contains indicators "
        "associated with impersonation scams. "
        "Voice analysis was not available. "
        "This is a risk estimate, not proof."
    ),

    "UNCERTAIN": (
        "Uncertain: some risk indicators were found, "
        "but voice analysis was not available. "
        "Please verify the caller independently."
    ),

    "LOW": (
        "Low risk based on call context only. "
        "Voice analysis was not available. "
        "This is an estimate, not a guarantee."
    ),
}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

_RECOMMENDATION = {
    "HIGH": (
        "Do not share OTP, PIN, passwords or banking "
        "information. Verify the caller through an "
        "independent trusted channel."
    ),

    "UNCERTAIN": (
        "Treat this call with caution. Do not share OTP, "
        "PIN, passwords or banking information until you "
        "have verified the caller through an independent "
        "trusted channel."
    ),

    "LOW": (
        "No strong risk indicators were found, but this "
        "is not a guarantee. Never share OTP, PIN or "
        "passwords on a call, and verify any unexpected "
        "request through a trusted channel."
    ),
}


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """
    Result returned by the risk engine.
    """

    assessment: RiskAssessment
    recommendation: str

    # Kept for compatibility with the existing backend.
    # Phone-number analysis is no longer used, so this is always 0.0.
    number_risk: float

    context_risk: float


# ---------------------------------------------------------------------------
# Main risk calculation
# ---------------------------------------------------------------------------

def assess_risk(
    synthetic_probability: Optional[float],
    financial_request: bool,
    identity_claim: Optional[str],
) -> RiskResult:
    """
    Calculate the final impersonation risk.

    Parameters
    ----------
    synthetic_probability:
        Probability from the AASIST voice deepfake model.
        Expected range: 0.0 - 1.0

    financial_request:
        Whether the caller requested money, OTP, PIN,
        banking information, etc.

    identity_claim:
        Identity the caller claims to have.

    Phone number is intentionally NOT accepted here.
    """

    # -------------------------------------------------------
    # 1. Determine whether ML voice analysis is available
    # -------------------------------------------------------

    voice_used = synthetic_probability is not None

    voice_risk = (
        synthetic_probability
        if voice_used
        else None
    )

    # -------------------------------------------------------
    # 2. Phone-number risk has been removed
    # -------------------------------------------------------

    number_risk = 0.0

    # -------------------------------------------------------
    # 3. Calculate call-context risk
    # -------------------------------------------------------

    context_risk = calculate_context_risk(
        financial_request=financial_request,
        identity_claim=identity_claim,
    )

    # -------------------------------------------------------
    # 4. Risk weights
    #
    # Voice       = 80%
    # Context     = 20%
    # Phone       = 0%
    # -------------------------------------------------------

    if voice_used:

        voice_weight = 0.80
        context_weight = 0.20

    else:

        # If the ML model is unavailable,
        # use the available context signal only.

        voice_weight = 0.0
        context_weight = 1.0

    number_weight = 0.0

    # -------------------------------------------------------
    # 5. Calculate individual contributions
    # -------------------------------------------------------

    voice_contribution = (
        (voice_risk or 0.0)
        * voice_weight
    )

    number_contribution = 0.0

    context_contribution = (
        context_risk
        * context_weight
    )

    # -------------------------------------------------------
    # 6. Calculate final score
    # -------------------------------------------------------

    score = (
        voice_contribution
        + number_contribution
        + context_contribution
    )

    # Keep score between 0 and 1.
    score = min(
        max(score, 0.0),
        1.0,
    )

    # Round to two decimal places.
    score = round(score, 2)

    # -------------------------------------------------------
    # 7. Convert score to LOW / UNCERTAIN / HIGH
    # -------------------------------------------------------

    level = score_to_level(score)

    # -------------------------------------------------------
    # 8. Generate explanations
    # -------------------------------------------------------

    reasons: list[str] = []

    # Voice explanation
    if not voice_used:

        reasons.append(
            "Voice analysis was unavailable, so this "
            "assessment uses call context only."
        )

    elif (
        synthetic_probability
        >= config.RISK_LEVEL_HIGH_MIN
    ):

        reasons.append(
            "Voice analysis indicates characteristics "
            "associated with synthetic speech."
        )

    elif (
        synthetic_probability
        >= config.RISK_LEVEL_UNCERTAIN_MIN
    ):

        reasons.append(
            "Voice analysis was inconclusive; some "
            "characteristics may be associated with "
            "synthetic speech."
        )

    else:

        reasons.append(
            "Voice analysis did not find strong indicators "
            "of synthetic speech."
        )

    # Financial request explanation
    if financial_request:

        reasons.append(
            "The conversation involves a financial request."
        )

    # Identity explanation
    if is_sensitive_identity(identity_claim):

        reasons.append(
            "The caller claims an identity that is commonly "
            "used in impersonation scams."
        )

    # -------------------------------------------------------
    # 9. Build the risk assessment
    # -------------------------------------------------------

    assessment = RiskAssessment(
        risk_score=score,

        risk_level=level,

        summary=(
            _LEVEL_SUMMARY
            if voice_used
            else _LEVEL_SUMMARY_NO_VOICE
        )[level],

        voice_analysis_used=voice_used,

        reasons=reasons,

        breakdown=RiskBreakdown(

            # -------------------------------
            # Voice
            # -------------------------------

            voice=ScoreComponent(
                risk=(
                    round(voice_risk, 4)
                    if voice_used
                    else None
                ),

                weight=round(
                    voice_weight,
                    4,
                ),

                contribution=round(
                    voice_contribution,
                    4,
                ),
            ),

            # -------------------------------
            # Number
            # -------------------------------

            number=ScoreComponent(
                risk=0.0,

                weight=0.0,

                contribution=0.0,
            ),

            # -------------------------------
            # Context
            # -------------------------------

            context=ScoreComponent(
                risk=round(
                    context_risk,
                    4,
                ),

                weight=round(
                    context_weight,
                    4,
                ),

                contribution=round(
                    context_contribution,
                    4,
                ),
            ),
        ),
    )

    # -------------------------------------------------------
    # 10. Return final result
    # -------------------------------------------------------

    return RiskResult(
        assessment=assessment,

        recommendation=_RECOMMENDATION[level],

        # Kept for compatibility with the existing
        # analysis service. Phone risk is disabled.
        number_risk=0.0,

        context_risk=context_risk,
    )
