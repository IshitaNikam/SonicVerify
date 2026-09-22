"""
Connect SonicVerify backend to the AASIST voice deepfake model.
"""

from inference import detect_voice as aasist_detect_voice


def detect_voice(audio_path: str) -> dict:
    """
    Run the AASIST model on the temporary audio file.
    """

    result = aasist_detect_voice(audio_path)

    spoof_probability = float(result["spoof_probability"])
    bonafide_probability = float(result["bonafide_probability"])

    confidence = max(
        spoof_probability,
        bonafide_probability
    )

    return {
        "synthetic_probability": spoof_probability,
        "authentic_probability": bonafide_probability,
        "confidence": confidence,
        "model_status": "available",
    }


def get_model_status() -> str:
    return "available"