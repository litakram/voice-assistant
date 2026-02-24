"""
main.py — Voice Assistant entry point.

Wires the four pipeline stages together (Capture → Transcribe → Brain → Speak)
and runs a continuous conversation loop with latency profiling.
"""

import argparse
import threading
import time

import config  # noqa: F401  — triggers .env loading & API key check
from capture import record_until_silence, numpy_to_wav_bytes
from transcribe import transcribe
from brain import ConversationBuffer, generate_response_stream, sentence_chunker
from speak import speak_stream, is_speaking

# ── Conversation state ──────────────────────────────────────────────
buffer = ConversationBuffer()


def run_once() -> None:
    """Execute one full listen → think → speak turn with latency profiling."""

    # Stage 1 — Capture
    print("\n🎙️  Listening…")
    t0 = time.perf_counter()
    audio = record_until_silence()
    t1 = time.perf_counter()

    if audio.size == 0:
        print("  ⚠️  No speech detected. Try again.")
        return

    wav_bytes = numpy_to_wav_bytes(audio)
    t2 = time.perf_counter()

    # Stage 2 — Transcribe
    text = transcribe(wav_bytes)
    t3 = time.perf_counter()

    if not text:
        print("  ⚠️  Could not transcribe. Try again.")
        return

    print(f"  📝 You said: {text}")
    buffer.add_user_message(text)

    # Stage 3 — Think
    print("  🧠 Thinking…")
    response_stream = generate_response_stream(buffer)
    sentences = sentence_chunker(response_stream)
    t4 = time.perf_counter()

    # Stage 4 — Speak
    speak_stream(sentences)
    t5 = time.perf_counter()

    # ── Latency report ──────────────────────────────────────────────
    print(
        f"\n  ⏱️  Latency breakdown:\n"
        f"      Capture …… {(t1 - t0) * 1000:>7.0f} ms\n"
        f"      WAV conv … {(t2 - t1) * 1000:>7.0f} ms\n"
        f"      Transcribe  {(t3 - t2) * 1000:>7.0f} ms\n"
        f"      Brain+TTS … {(t5 - t4) * 1000:>7.0f} ms\n"
        f"      Total ————— {(t5 - t0) * 1000:>7.0f} ms"
    )


# ── Barge-in listener (background thread) ──────────────────────────

def _barge_in_listener() -> None:
    """
    Run energy-based VAD on microphone input while the assistant is speaking.
    If voice activity is detected, clear `is_speaking` to interrupt
    playback so the user can take over.
    """
    import sounddevice as sd
    from capture import SAMPLE_RATE, FRAME_SAMPLES, FRAME_DURATION_MS, _is_speech, ENERGY_THRESHOLD

    speech_frames = 0
    TRIGGER_FRAMES = 5  # ~150 ms of speech to trigger barge-in

    def _cb(indata, frame_count, time_info, status):
        nonlocal speech_frames
        if not is_speaking.is_set():
            speech_frames = 0
            return

        frame = indata[:, 0].copy()
        if _is_speech(frame, ENERGY_THRESHOLD):
            speech_frames += 1
            if speech_frames >= TRIGGER_FRAMES:
                is_speaking.clear()
                speech_frames = 0
        else:
            speech_frames = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=_cb,
    ):
        while True:
            time.sleep(FRAME_DURATION_MS / 1000)


# ── CLI argument parsing ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="🎙️ Voice Assistant — Speech-to-Speech AI Pipeline"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        help="TTS voice to use (overrides .env / default).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chat model to use, e.g. gpt-4o, gpt-4o-mini (overrides .env).",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────

def main():
    args = parse_args()

    # CLI overrides
    if args.voice:
        config.TTS_VOICE = args.voice
    if args.model:
        config.GPT_MODEL = args.model

    print("=" * 50)
    print("  🎙️  Voice Assistant")
    print(f"  Model : {config.GPT_MODEL}")
    print(f"  Voice : {config.TTS_VOICE}")
    print("  Press Ctrl+C to quit")
    print("=" * 50)

    # Start barge-in listener in background
    barge_thread = threading.Thread(target=_barge_in_listener, daemon=True)
    barge_thread.start()

    try:
        while True:
            run_once()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
