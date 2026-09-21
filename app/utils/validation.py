"""
Input validation helpers:
  * normalize_phone_number()  - Indian numbers -> +91XXXXXXXXXX
  * validate_audio_upload()   - size / extension / basic integrity checks

Audio integrity checks use only the standard library (magic bytes + container
structure), so no ffmpeg or audio libraries are needed for the backend.
"""
import re
import struct
from pathlib import Path

from app import config
from app.exceptions import (
    CorruptedAudioError,
    EmptyFileError,
    FileTooLargeError,
    InvalidPhoneNumberError,
    UnsupportedAudioFormatError,
)

# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------
_ALLOWED_PHONE_CHARS = re.compile(r"\+?[\d\s\-().]+")
_INDIAN_MOBILE = re.compile(r"[6-9]\d{9}")


def normalize_phone_number(phone_number) -> str:
    """
    Normalize an Indian mobile number to +91XXXXXXXXXX.

    9876543210, 09876543210, 919876543210, +91 98765-43210 and 00919876543210
    all become +919876543210. Anything else raises InvalidPhoneNumberError.
    """
    if phone_number is None or not str(phone_number).strip():
        raise InvalidPhoneNumberError("Phone number is required.")

    raw = str(phone_number).strip()
    if not _ALLOWED_PHONE_CHARS.fullmatch(raw):
        raise InvalidPhoneNumberError(
            "Phone number may only contain digits, spaces, '-', '(', ')' and an optional leading '+'."
        )

    digits = re.sub(r"\D", "", raw)
    if digits.startswith("0091"):  # 00 = international dialling prefix
        digits = digits[2:]

    if len(digits) == 10:
        national = digits
    elif len(digits) == 11 and digits.startswith("0"):  # trunk prefix
        national = digits[1:]
    elif len(digits) == 12 and digits.startswith("91"):
        national = digits[2:]
    else:
        raise InvalidPhoneNumberError(
            "Invalid phone number. This prototype supports 10-digit Indian mobile numbers "
            "(e.g. 9876543210, 919876543210 or +919876543210)."
        )

    if not _INDIAN_MOBILE.fullmatch(national) or len(set(national)) == 1:
        raise InvalidPhoneNumberError(
            "Invalid Indian mobile number. It must have 10 digits and start with 6, 7, 8 or 9."
        )
    return f"+91{national}"


# ---------------------------------------------------------------------------
# Audio uploads
# ---------------------------------------------------------------------------
def validate_audio_upload(filename: str | None, data: bytes) -> str:
    """
    Validate an uploaded audio file. Returns the file extension (".wav", ".mp3"
    or ".m4a") or raises an AppError subclass.
    """
    if not data:
        raise EmptyFileError()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise FileTooLargeError(f"File too large. Maximum allowed size is {config.MAX_UPLOAD_MB} MB.")

    ext = Path(filename or "").suffix.lower()
    if ext:
        if ext not in config.ALLOWED_AUDIO_EXTENSIONS:
            raise UnsupportedAudioFormatError()
    else:
        # No extension (e.g. some browser/Streamlit recordings): detect from content.
        ext = _sniff_extension(data)
        if ext is None:
            raise UnsupportedAudioFormatError()

    {".wav": _validate_wav, ".mp3": _validate_mp3, ".m4a": _validate_m4a}[ext](data)
    return ext


def _sniff_extension(data: bytes) -> str | None:
    if data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE":
        return ".wav"
    if data[4:8] == b"ftyp":
        return ".m4a"
    if data[:3] == b"ID3" or (len(data) > 3 and _is_mp3_frame_header(data[:4])):
        return ".mp3"
    return None


def _validate_wav(data: bytes) -> None:
    """Walk the RIFF chunks; require a sane 'fmt ' chunk and a non-empty 'data' chunk."""
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise CorruptedAudioError()

    pos, n = 12, len(data)
    fmt_ok = False
    while pos + 8 <= n:
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8

        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > n:
                raise CorruptedAudioError()
            _fmt, channels, sample_rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if channels < 1 or sample_rate < 1 or bits < 1:
                raise CorruptedAudioError()
            fmt_ok = True
        elif chunk_id == b"data":
            if not fmt_ok:
                raise CorruptedAudioError()
            available = n - body
            if available <= 0:
                raise CorruptedAudioError("The WAV file contains no audio data.")
            # size 0 / 0xFFFFFFFF = "unknown length" written by some streaming recorders
            if size not in (0, 0xFFFFFFFF) and size > available:
                raise CorruptedAudioError("The WAV file is truncated.")
            return

        pos = body + size + (size & 1)  # chunks are padded to even sizes

    raise CorruptedAudioError()


def _is_mp3_frame_header(h: bytes) -> bool:
    if len(h) < 4 or h[0] != 0xFF or (h[1] & 0xE0) != 0xE0:
        return False
    version = (h[1] >> 3) & 3
    layer = (h[1] >> 1) & 3
    bitrate_idx = (h[2] >> 4) & 0xF
    sample_rate_idx = (h[2] >> 2) & 3
    return version != 1 and layer != 0 and bitrate_idx != 0xF and sample_rate_idx != 3


def _validate_mp3(data: bytes) -> None:
    """Skip an optional ID3v2 tag, then require a valid MPEG frame header nearby."""
    pos = 0
    if data[:3] == b"ID3":
        if len(data) < 10:
            raise CorruptedAudioError()
        tag_size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        pos = 10 + tag_size
        if pos >= len(data):
            raise CorruptedAudioError()

    end = min(len(data) - 3, pos + 65536)
    i = pos
    while i < end:
        j = data.find(b"\xff", i, end)
        if j == -1:
            break
        if _is_mp3_frame_header(data[j:j + 4]):
            return
        i = j + 1
    raise CorruptedAudioError()


def _validate_m4a(data: bytes) -> None:
    """Walk the top-level MP4 boxes; a playable M4A needs both 'moov' and 'mdat'."""
    if len(data) < 12 or data[4:8] != b"ftyp":
        raise CorruptedAudioError()

    pos, n = 0, len(data)
    seen: set[bytes] = set()
    while pos + 8 <= n:
        size = int.from_bytes(data[pos:pos + 4], "big")
        box_type = data[pos + 4:pos + 8]
        header = 8
        if size == 1:  # 64-bit size follows
            if pos + 16 > n:
                break
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:  # box extends to end of file
            size = n - pos
        if size < header:
            raise CorruptedAudioError()
        seen.add(box_type)
        pos += size

    if b"moov" not in seen or b"mdat" not in seen:
        raise CorruptedAudioError()
