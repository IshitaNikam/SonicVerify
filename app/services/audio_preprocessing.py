import librosa
import numpy as np


# Basic audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 5
N_MFCC = 40


def preprocess_audio(file):

    # Load audio and convert it to mono at 16 kHz
    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize the audio volume
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    # Store 5-second audio chunks
    chunks = []

    samples_per_chunk = sr * CHUNK_SIZE

    for start in range(0, len(audio), samples_per_chunk):

        chunk = audio[start:start + samples_per_chunk]

        # Ignore very short chunks
        if len(chunk) >= sr:
            chunks.append(chunk)

    # Extract MFCC features
    mfcc_features = []

    for chunk in chunks:

        mfcc = librosa.feature.mfcc(
            y=chunk,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_features.append(mfcc)

    return chunks, mfcc_features

