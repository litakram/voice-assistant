"""
test_pipeline.py — Isolated tests for each pipeline stage.

Run individual tests:
    python test_pipeline.py capture
    python test_pipeline.py transcribe
    python test_pipeline.py brain
    python test_pipeline.py speak

Or run all:
    python test_pipeline.py all
"""

import sys
import time


def test_capture():
    """Stage 1: Record 3 seconds of audio and print array shape."""
    import numpy as np
    import sounddevice as sd
    from capture import SAMPLE_RATE, numpy_to_wav_bytes

    print("🎤 Recording 3 seconds of audio…")
    duration = 3  # seconds
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    audio = audio.flatten()

    print(f"   ✅ Array shape : {audio.shape}")
    print(f"   ✅ Dtype       : {audio.dtype}")
    print(f"   ✅ Duration    : {len(audio) / SAMPLE_RATE:.2f}s")

    # Also test WAV conversion
    wav = numpy_to_wav_bytes(audio)
    print(f"   ✅ WAV size    : {wav.getbuffer().nbytes:,} bytes")
    return audio, wav


def test_transcribe(wav_bytes=None):
    """Stage 2: Transcribe a WAV (from Stage 1 or a hardcoded test)."""
    from transcribe import transcribe

    if wav_bytes is None:
        # Record a quick clip for testing
        print("   ℹ️  No WAV provided — recording 3s for transcription test…")
        _, wav_bytes = test_capture()

    print("📝 Sending to Whisper…")
    t0 = time.perf_counter()
    text = transcribe(wav_bytes)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"   ✅ Transcript  : \"{text}\"")
    print(f"   ✅ Latency     : {elapsed:.0f} ms")
    return text


def test_brain(user_text=None):
    """Stage 3: Send text to the brain and print streamed response."""
    from brain import ConversationBuffer, generate_response_stream, sentence_chunker

    if user_text is None:
        user_text = "Tell me a fun fact about octopuses."

    print(f"🧠 Sending to brain: \"{user_text}\"")

    buf = ConversationBuffer()
    buf.add_user_message(user_text)

    t0 = time.perf_counter()
    stream = generate_response_stream(buf)

    print("   Response (streamed): ", end="", flush=True)
    full = ""
    for chunk in stream:
        print(chunk, end="", flush=True)
        full += chunk
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n   ✅ Length       : {len(full)} chars")
    print(f"   ✅ Latency     : {elapsed:.0f} ms")

    # Test sentence chunker
    buf2 = ConversationBuffer()
    buf2.add_user_message(user_text)
    stream2 = generate_response_stream(buf2)
    sentences = list(sentence_chunker(stream2))
    print(f"   ✅ Sentences   : {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"      {i}. {s}")

    return full


def test_speak():
    """Stage 4: Synthesize a short sentence and play it."""
    from speak import synthesize, play_audio

    sentence = "Hello! I am your voice assistant. Nice to meet you."
    print(f"🔊 Synthesizing: \"{sentence}\"")

    t0 = time.perf_counter()
    audio = synthesize(sentence)
    t_synth = (time.perf_counter() - t0) * 1000
    print(f"   ✅ TTS latency : {t_synth:.0f} ms")
    print(f"   ✅ Audio size  : {audio.getbuffer().nbytes:,} bytes")

    print("   ▶️  Playing…")
    t0 = time.perf_counter()
    play_audio(audio)
    t_play = (time.perf_counter() - t0) * 1000
    print(f"   ✅ Playback    : {t_play:.0f} ms")


# ── CLI dispatcher ──────────────────────────────────────────────────

TESTS = {
    "capture": test_capture,
    "transcribe": test_transcribe,
    "brain": test_brain,
    "speak": test_speak,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <capture|transcribe|brain|speak|all>")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "all":
        for name, fn in TESTS.items():
            print(f"\n{'=' * 50}")
            print(f"  Testing: {name}")
            print(f"{'=' * 50}")
            fn()
    elif target in TESTS:
        TESTS[target]()
    else:
        print(f"Unknown test: {target}")
        print(f"Available: {', '.join(TESTS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
