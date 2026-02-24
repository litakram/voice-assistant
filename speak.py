"""
speak.py — Stage 4: Text-to-Speech + Playback.

Synthesizes sentences via the OpenAI TTS API and plays them through
the default speakers using sounddevice.
Uses raw PCM output to avoid FFmpeg dependency.
"""

import io
import threading
from typing import Generator

import numpy as np
import sounddevice as sd
from openai import OpenAI

import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Shared event: set while audio is actively playing
is_speaking = threading.Event()


def synthesize(text: str) -> io.BytesIO:
    """
    Send a text string to the OpenAI TTS API and return raw PCM audio.

    Parameters
    ----------
    text : str
        The sentence to synthesize.

    Returns
    -------
    io.BytesIO
        In-memory PCM audio data (24kHz, 16-bit mono).
    """
    response = _client.audio.speech.create(
        model="tts-1",
        voice=config.TTS_VOICE,
        input=text,
        response_format="pcm",  # Raw PCM instead of MP3
    )
    buf = io.BytesIO(response.content)
    buf.seek(0)
    return buf


def play_audio(audio_bytes: io.BytesIO) -> None:
    """
    Play raw PCM audio through the default speakers.

    Blocks until playback is complete.

    Parameters
    ----------
    audio_bytes : io.BytesIO
        In-memory PCM data (24kHz, 16-bit mono).
    """
    audio_bytes.seek(0)
    # Convert raw bytes to numpy array (16-bit signed integer)
    samples = np.frombuffer(audio_bytes.read(), dtype=np.int16)

    # TTS "pcm" format is 24,000 Hz, mono
    sd.play(samples, samplerate=24_000)
    sd.wait()  # block until done


def speak_stream(
    sentence_stream: Generator[str, None, None],
) -> None:
    """
    Iterate over sentence strings, synthesize each one, and play it
    immediately — producing a natural, sentence-by-sentence speaking
    cadence without waiting for the full response.

    Respects the `is_speaking` event for barge-in support: if the
    event is cleared externally, remaining sentences are skipped.

    Parameters
    ----------
    sentence_stream : Generator[str, None, None]
        A generator yielding one sentence at a time.
    """
    is_speaking.set()

    try:
        for sentence in sentence_stream:
            if not is_speaking.is_set():
                if config.DEBUG:
                    print("  🛑 Barge-in: playback interrupted.")
                break

            if config.DEBUG:
                print(f"  🔊 Speaking: {sentence}")

            audio = synthesize(sentence)
            play_audio(audio)
    finally:
        is_speaking.clear()
