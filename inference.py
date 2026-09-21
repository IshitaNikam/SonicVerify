
import numpy as np
import librosa
import torch
from pathlib import Path

from AASIST import Model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "AASIST.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = Model(CONFIG).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()


# --------------------------------------------------
# Voice deepfake detection
# --------------------------------------------------

def detect_voice(audio_path):
    """
    Detect whether an audio file is BONAFIDE or SPOOF.

    Class 0 = SPOOF
    Class 1 = BONAFIDE

    Returns a dictionary containing the final prediction,
    probabilities, number of chunks, and individual
    chunk results.
    """

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    chunk_size = 64600

    chunks = []

    for start in range(0, len(audio), chunk_size):

        chunk = audio[start:start + chunk_size]

        if len(chunk) == 0:
            continue

        if len(chunk) < chunk_size:

            chunk = np.tile(
                chunk,
                int(np.ceil(
                    chunk_size / len(chunk)
                ))
            )[:chunk_size]

        chunks.append(chunk)

    if not chunks:
        raise ValueError("Audio file is empty.")

    chunk_results = []

    for i, chunk in enumerate(chunks):

        x = torch.tensor(
            chunk,
            dtype=torch.float32
        )

        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            _, output = model(x)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        spoof_probability = probabilities[0, 0].item()
        bonafide_probability = probabilities[0, 1].item()

        prediction = (
            "BONAFIDE"
            if bonafide_probability >= 0.5
            else "SPOOF"
        )

        chunk_results.append({
            "chunk": i + 1,
            "spoof_probability": spoof_probability,
            "bonafide_probability": bonafide_probability,
            "prediction": prediction
        })


    average_spoof = float(np.mean([
        r["spoof_probability"]
        for r in chunk_results
    ]))

    average_bonafide = float(np.mean([
        r["bonafide_probability"]
        for r in chunk_results
    ]))

    final_prediction = (
        "BONAFIDE"
        if average_bonafide >= 0.5
        else "SPOOF"
    )


    return {
        "prediction": final_prediction,
        "spoof_probability": average_spoof,
        "bonafide_probability": average_bonafide,
        "number_of_chunks": len(chunks),
        "chunks": chunk_results
    }
