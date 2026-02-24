"""
speak.py — Stage 4: Text-to-Speech + Playback.

Synthesizes sentences via the OpenAI TTS API and plays them through
the default speakers using sounddevice + pydub.
"""

import io
import threading
from typing import Generator

import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from openai import OpenAI

import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Shared event: set while audio is actively playing
is_speaking = threading.Event()


def synthesize(text: str) -> io.BytesIO:
    """
    Send a text string to the OpenAI TTS API and return audio bytes.

    Parameters
    ----------
    text : str
        The sentence to synthesize.

    Returns
    -------
    io.BytesIO
        In-memory MP3 audio data.
    """
    response = _client.audio.speech.create(
        model="tts-1",
        voice=config.TTS_VOICE,
        input=text,
    )
    buf = io.BytesIO(response.content)
    buf.seek(0)
    return buf


def play_audio(audio_bytes: io.BytesIO) -> None:
    """
    Decode MP3 audio and play it through the default speakers.

    Uses pydub for MP3 → PCM decoding and sounddevice for playback.
    Blocks until playback is complete.

    Parameters
    ----------
    audio_bytes : io.BytesIO
        In-memory MP3 data from the TTS API.
    """
    audio_bytes.seek(0)
    segment = AudioSegment.from_mp3(audio_bytes)

    # Convert to raw PCM numpy array
    samples = np.array(segment.get_array_of_samples(), dtype=np.int16)

    # Handle stereo → reshape
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))

    sd.play(samples, samplerate=segment.frame_rate)
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
