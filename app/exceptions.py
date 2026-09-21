"""Application errors. Each one maps to a clean JSON error response in main.py."""


class AppError(Exception):
    status_code = 400
    default_message = "Bad request."

    def __init__(self, message: str | None = None, status_code: int | None = None):
        self.message = message or self.default_message
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)


class InvalidPhoneNumberError(AppError):
    status_code = 400
    default_message = "Invalid phone number."


class EmptyFileError(AppError):
    status_code = 400
    default_message = "The uploaded audio file is empty."


class UnsupportedAudioFormatError(AppError):
    status_code = 415
    default_message = "Unsupported audio format. Please upload WAV, MP3 or M4A."


class FileTooLargeError(AppError):
    status_code = 413
    default_message = "The uploaded file is too large."


class CorruptedAudioError(AppError):
    status_code = 400
    default_message = "The audio file appears to be corrupted or is not a valid WAV, MP3 or M4A file."
