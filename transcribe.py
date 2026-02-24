"""
transcribe.py — Stage 2: Speech-to-Text.

Sends an in-memory WAV byte stream to the OpenAI Whisper API and
returns the transcribed text.
"""

import io
from openai import OpenAI
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def transcribe(audio_bytes: io.BytesIO) -> str:
    """
    Transcribe speech from an in-memory WAV file using OpenAI Whisper.

    Parameters
    ----------
    audio_bytes : io.BytesIO
        A BytesIO object containing WAV audio data (must have a `.name`
        attribute ending in `.wav`).

    Returns
    -------
    str
        The transcribed text, or an empty string on failure.
    """
    try:
        result = _client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
        )
        text = result.text.strip()

        if config.DEBUG:
            print(f"  🔤 Transcription: {text}")

        return text

    except Exception as exc:
        if config.DEBUG:
            print(f"  ⚠️  Whisper API error: {exc}")
        return ""
