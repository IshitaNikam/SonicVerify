# Voice Deepfake Detection - AASIST

## Files

- AASIST.py - AASIST model architecture
- AASIST.pth - Pretrained model weights
- inference.py - Voice detection function
- requirements.txt - Required libraries

## Model Input

Audio is converted to:
- 16 kHz
- Mono
- Approximately 4-second chunks
- 64,600 samples per chunk

## Classes

Class 0 = SPOOF
Class 1 = BONAFIDE

## Usage

Install dependencies:

pip install -r requirements.txt

Then use:

from inference import detect_voice

result = detect_voice("audio.wav")

print(result["prediction"])
print(result["spoof_probability"])
print(result["bonafide_probability"])

## Integration

The application team can pass the recorded audio file path to:

detect_voice(audio_path)

The returned prediction can be displayed in the application.

## Note

The model was primarily evaluated using ASVspoof 2019 Logical Access data.
Performance on new languages, accents, microphones, and real-world
recordings may differ.
