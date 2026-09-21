"""
ML INTEGRATION POINT  (owned by the ML teammate)
================================================
The backend only needs this module to expose two functions:

    detect_voice(audio_path: str) -> dict
    get_model_status() -> str

`detect_voice` receives the path of a temporary audio file (.wav/.mp3/.m4a) and
must return a dict with exactly these keys:

    {
        "synthetic_probability": 0.87,   # float 0-1: P(voice is AI-generated / cloned)
        "authentic_probability": 0.13,   # float 0-1: P(voice is a real human)
        "confidence": 0.87,              # float 0-1: model confidence in its prediction
        "model_status": "available",
    }

If the model cannot run, return model_status "unavailable" (probabilities may be
None). Do NOT return made-up numbers. Exceptions are also handled by the backend
(treated as model_status "error"), so nothing here can crash the API.

To plug in the real model, replace the body of the two functions below - no API
route or other backend file has to change. Example:

    from ml.voice_model import load_model, predict     # teammate's code
    _model = load_model()

    def detect_voice(audio_path):
        p_fake = float(predict(_model, audio_path))
        return {"synthetic_probability": p_fake,
                "authentic_probability": 1 - p_fake,
                "confidence": max(p_fake, 1 - p_fake),
                "model_status": "available"}

    def get_model_status():
        return "available"
"""


def detect_voice(audio_path: str) -> dict:
    """PLACEHOLDER - no real model is connected yet, so this reports 'unavailable'."""
    return {
        "synthetic_probability": None,
        "authentic_probability": None,
        "confidence": None,
        "model_status": "unavailable",
    }


def get_model_status() -> str:
    """Used by /api/health. Return "available" once the real model is loaded."""
    return "unavailable"
