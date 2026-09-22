import os

import librosa
import numpy as np
import soundfile as sf


# Basic audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 4
MIN_CHUNK_DURATION = 1

# Silence detection settings
TOP_DB = 30
SILENCE_THRESHOLD = 0.01


def preprocess_audio(file):
    """
    Preprocess audio and split it into 4-second chunks.

    Returns:
        chunks: list of audio arrays
        sample_rate: audio sample rate
    """

    if not os.path.exists(file):
        raise FileNotFoundError("Audio file not found.")

    # Load audio as mono at 16 kHz
    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file contains no usable audio.")

    # Normalize audio volume
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    # Remove silence from the beginning and end
    audio, _ = librosa.effects.trim(
        audio,
        top_db=TOP_DB
    )

    if len(audio) == 0:
        raise ValueError("No speech detected in the audio.")

    # Create 4-second chunks
    chunks = []
    samples_per_chunk = sr * CHUNK_SIZE

    for start in range(0, len(audio), samples_per_chunk):

        end = start + samples_per_chunk
        chunk = audio[start:end]

        # Ignore very short final chunks
        if len(chunk) < sr * MIN_CHUNK_DURATION:
            continue

        # Skip nearly silent chunks
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms < SILENCE_THRESHOLD:
            continue

        # Pad final chunk to 4 seconds
        if len(chunk) < samples_per_chunk:
            padding = samples_per_chunk - len(chunk)

            chunk = np.pad(
                chunk,
                (0, padding),
                mode="constant"
            )

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable speech chunks found.")

    return chunks, sr


def save_chunks(chunks, output_folder="audio_chunks"):
    """
    Save processed chunks as WAV files.
    """

    os.makedirs(output_folder, exist_ok=True)

    chunk_paths = []

    for index, chunk in enumerate(chunks):

        file_path = os.path.join(
            output_folder,
            f"chunk_{index + 1}.wav"
        )

        # Save as 16-bit PCM WAV
        sf.write(
            file_path,
            chunk,
            SAMPLE_RATE,
            subtype="PCM_16"
        )

        chunk_paths.append(file_path)

    return chunk_paths