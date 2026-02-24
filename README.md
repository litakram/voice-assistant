# 🎙️ Voice Assistant — Speech-to-Speech AI Pipeline

A real-time, conversational voice assistant that listens, transcribes,
thinks, and speaks — all in a streaming pipeline.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  🎤 Capture │───▶│  📝 Whisper │───▶│  🧠 GPT-4o  │───▶│  🔊 TTS     │
│  sounddevice│    │  STT API    │    │  Streaming   │    │  + Playback │
│  + webrtcvad│    │             │    │  + Chunker   │    │  + pydub    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     Audio              Text              Text              Audio
   (16 kHz)          (transcript)      (sentences)        (MP3 → PCM)
```

## Features

- **Voice Activity Detection** — starts/stops recording automatically
- **Streaming responses** — speaks sentence-by-sentence as GPT generates
- **Barge-in support** — interrupt the assistant by speaking
- **Conversation memory** — rolling context window
- **Latency profiling** — per-stage timing after each turn
- **CLI overrides** — `--voice` and `--model` flags

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — required by pydub for MP3 decoding
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **OpenAI API key** with access to Whisper, Chat Completions, and TTS APIs
- **Microphone + speakers** connected to your system

## Setup

1. **Clone / navigate to the project:**
   ```bash
   cd voice_assistant
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and paste your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-real-key-here
   ```

## Usage

**Start the assistant:**
```bash
python main.py
```

**With CLI overrides:**
```bash
python main.py --voice shimmer --model gpt-4o
```

**Available voices:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

**Stop:** Press `Ctrl+C` to exit gracefully.

## Testing

Test each stage independently:
```bash
python test_pipeline.py capture      # Record 3s and print array shape
python test_pipeline.py transcribe   # Record + send to Whisper
python test_pipeline.py brain        # Send text to GPT and stream response
python test_pipeline.py speak        # Synthesize and play a sentence
python test_pipeline.py all          # Run all tests in sequence
```

## Environment Variables

| Variable         | Default       | Description                    |
| ---------------- | ------------- | ------------------------------ |
| `OPENAI_API_KEY` | *(required)*  | Your OpenAI API key            |
| `TTS_VOICE`      | `alloy`       | TTS voice name                 |
| `GPT_MODEL`      | `gpt-4o-mini` | Chat model                     |
| `DEBUG`          | `false`       | Print debug info to console    |

## Project Structure

```
voice_assistant/
├── main.py             # Entry point & conversation loop
├── capture.py          # Microphone recording + VAD
├── transcribe.py       # Whisper STT
├── brain.py            # GPT chat + sentence chunker
├── speak.py            # TTS + audio playback
├── config.py           # Environment / config loader
├── test_pipeline.py    # Per-stage tests
├── requirements.txt    # Pinned dependencies
├── .env.example        # Environment template
└── README.md           # This file
```
