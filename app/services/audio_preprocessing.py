import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


# Standard audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

# Silence detection settings
TOP_DB = 30
SILENCE_THRESHOLD = 0.01


def convert_to_wav(input_file, output_file):
    """Convert input audio to standard WAV."""

    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ac", "1",                  # Mono
        "-ar", str(SAMPLE_RATE),     # 16 kHz
        "-c:a", "pcm_s16le",         # 16-bit PCM
        output_file
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise ValueError("Invalid or unsupported audio file.")


def preprocess_audio(input_file, output_folder="audio_chunks"):
    """
    Preprocess audio and return WAV chunk paths.
    """

    if not os.path.isfile(input_file):
        raise FileNotFoundError("Audio file not found.")

    os.makedirs(output_folder, exist_ok=True)

    # Temporary standardized WAV
    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    try:
        # Convert input to WAV
        convert_to_wav(input_file, temp_wav)

        # Load standardized audio
        audio, _ = librosa.load(
            temp_wav,
            sr=SAMPLE_RATE,
            mono=True
        )

        if audio.size == 0:
            raise ValueError("Audio contains no usable data.")

        # Normalize volume
        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = audio / peak

        # Trim silence at beginning and end
        audio, _ = librosa.effects.trim(
            audio,
            top_db=TOP_DB
        )

        if audio.size == 0:
            raise ValueError("No usable speech found.")

        chunk_paths = []

        # Create 4-second chunks
        for start in range(0, len(audio), CHUNK_SAMPLES):

            end = start + CHUNK_SAMPLES
            chunk = audio[start:end]

            # Skip chunks shorter than 1 second
            if len(chunk) < SAMPLE_RATE:
                continue

            # Skip nearly silent chunks
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms < SILENCE_THRESHOLD:
                continue

            # Pad final chunk to 4 seconds
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(
                    chunk,
                    (0, CHUNK_SAMPLES - len(chunk))
                )

            # Save as 16-bit WAV
            chunk_path = os.path.join(
                output_folder,
                f"chunk_{len(chunk_paths) + 1}.wav"
            )

            sf.write(
                chunk_path,
                chunk,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            chunk_paths.append(chunk_path)

        if not chunk_paths:
            raise ValueError("No usable audio chunks found.")

        return chunk_paths

    finally:
        # Remove temporary WAV
        if os.path.exists(temp_wav):
            os.remove(temp_wav)