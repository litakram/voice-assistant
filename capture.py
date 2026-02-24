"""
capture.py — Stage 1: Audio Capture.

Records audio from the default microphone at 16 kHz using sounddevice,
applies energy-based VAD to detect speech, and returns the recording as a
NumPy array or as in-memory WAV bytes ready for the Whisper API.

Uses pure-Python energy-based voice activity detection — no C compiler needed.
"""

import io
import wave
import numpy as np
import sounddevice as sd
import config

# ── Constants ───────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz
CHANNELS = 1                  # mono
DTYPE = "int16"               # 16-bit PCM
FRAME_DURATION_MS = 30        # 30 ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

SILENCE_TIMEOUT_S = 1.5       # seconds of silence before we stop recording
PRE_SPEECH_BUFFER = 15        # number of frames to keep before speech starts
MIN_RECORDING_MS = 500        # Minimum recording duration (Avoid tiny clips)

SILENCE_FRAMES_THRESHOLD = int(SILENCE_TIMEOUT_S * 1000 / FRAME_DURATION_MS)
MIN_FRAMES_THRESHOLD = int(MIN_RECORDING_MS / FRAME_DURATION_MS)

# Energy-based VAD thresholds
ENERGY_THRESHOLD = 300        # Default RMS threshold (will be auto-calibrated)
CALIBRATION_FRAMES = 30       # frames to use for background noise calibration


def _rms(frame: np.ndarray) -> float:
    """Compute root-mean-square energy of an audio frame."""
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def _is_speech(frame: np.ndarray, threshold: float) -> bool:
    """Return True if frame energy exceeds the speech threshold."""
    return _rms(frame) > threshold


def record_until_silence() -> np.ndarray:
    """
    Record from the microphone until speech is detected and then
    1.5 seconds of continuous silence follows.

    Uses energy-based voice activity detection (no external C libs).

    Returns
    -------
    np.ndarray
        Recorded audio as a 1-D int16 NumPy array at 16 kHz.
    """
    frames: list[np.ndarray] = []
    ring_buffer: list[np.ndarray] = []   # pre-speech buffer
    recording = False
    silent_frame_count = 0
    calibration_energies: list[float] = []
    threshold = ENERGY_THRESHOLD

    def _callback(indata: np.ndarray, frame_count: int, time_info, status):
        nonlocal recording, silent_frame_count, threshold

        if status:
            if config.DEBUG:
                print(f"  ⚠️ Sounddevice Status: {status}")

        frame = indata[:, 0].copy()
        current_energy = _rms(frame)

        # Auto-calibrate threshold from background noise
        if len(calibration_energies) < CALIBRATION_FRAMES:
            calibration_energies.append(current_energy)
            if len(calibration_energies) == CALIBRATION_FRAMES:
                noise_floor = np.mean(calibration_energies)
                # Set threshold to 3x noise floor or at least 300
                threshold = max(ENERGY_THRESHOLD, noise_floor * 3)
                if config.DEBUG:
                    print(f"  📶 VAD Calibrated: Noise Floor: {noise_floor:.1f}, Threshold: {threshold:.1f}")
            return

        is_voice = current_energy > threshold

        if not recording:
            ring_buffer.append(frame.copy())
            if len(ring_buffer) > PRE_SPEECH_BUFFER:
                ring_buffer.pop(0)

            if is_voice:
                recording = True
                silent_frame_count = 0
                frames.extend(ring_buffer)
                ring_buffer.clear()
                if config.DEBUG:
                    print(f"  🔊 Voice detected (Energy: {current_energy:.1f})")
        else:
            frames.append(frame.copy())
            if is_voice:
                silent_frame_count = 0
            else:
                silent_frame_count += 1
                if config.DEBUG and silent_frame_count % 10 == 0:
                    print(f"  🤫 Silence... ({silent_frame_count}/{SILENCE_FRAMES_THRESHOLD})")

    # Open an InputStream and block until silence timeout
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=FRAME_SAMPLES,
            callback=_callback,
        ):
            import time
            start_time = time.time()
            while True:
                time.sleep(FRAME_DURATION_MS / 1000)
                # Stop if silence threshold reached AND we have enough audio
                if recording and silent_frame_count >= SILENCE_FRAMES_THRESHOLD:
                    if len(frames) >= MIN_FRAMES_THRESHOLD:
                        break
                    else:
                        # Reset and wait for more speech if recording was too short
                        if config.DEBUG:
                            print("  ⚠️ Recording too short, waiting for more speech...")
                        recording = False
                        frames.clear()
                
                # Global timeout to prevent infinite hanging (e.g. 30s)
                if time.time() - start_time > 30:
                    if config.DEBUG:
                        print("  🛑 Recording timed out (30s).")
                    break

    except Exception as e:
        print(f"  ❌ Microphone Error: {e}")
        return np.array([], dtype=np.int16)

    if not frames:
        return np.array([], dtype=np.int16)

    return np.concatenate(frames).astype(np.int16)


def numpy_to_wav_bytes(audio_array: np.ndarray) -> io.BytesIO:
    """
    Convert a 1-D int16 NumPy array to an in-memory WAV byte stream.

    Parameters
    ----------
    audio_array : np.ndarray
        Audio samples at 16 kHz, int16.

    Returns
    -------
    io.BytesIO
        A seeked-to-zero BytesIO object containing valid WAV data,
        ready to be sent directly to the Whisper API.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_array.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"   # Whisper API needs a filename hint
    return buf
