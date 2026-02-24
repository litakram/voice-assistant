<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Whisper-STT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TTS-Streaming-orange?style=for-the-badge" />
</p>

<h1 align="center">🎙️ Voice Assistant</h1>
<h3 align="center"><em>Talk. Listen. Think. Speak — All in real time.</em></h3>

<p align="center">
  A blazing-fast, speech-to-speech conversational AI pipeline that captures your voice,<br/>
  transcribes it with Whisper, thinks with GPT-4o, and speaks back — sentence by sentence.
</p>

---

## ⚡ How It Works

```
     YOU speak                                        ASSISTANT replies
         │                                                  ▲
         ▼                                                  │
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  🎤  Capture   │─▶│ 📝  Transcribe  │─▶│    🧠  Brain    │─▶│   🔊  Speak     │
│  sounddevice    │  │  Whisper STT     │  │  GPT-4o          │  │  TTS + Playback  │
│  energy-based   │  │  API             │  │  streaming       │  │  raw PCM         │
│  VAD            │  │                  │  │  + sentence      │  │  via sounddevice │
│                 │  │                  │  │    chunker       │  │                  │
└─────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
   16 kHz mono           transcript           sentences             24 kHz audio
```

---

## ✨ Features

| Feature | Description |
|---------|------------|
| 🗣️ **Auto Voice Detection** | Starts & stops recording automatically using energy-based VAD with noise calibration |
| ⚡ **Streaming Responses** | Speaks sentence-by-sentence as GPT generates — no waiting for full response |
| 🛑 **Barge-in** | Interrupt the assistant mid-sentence just by speaking |
| 🧠 **Conversation Memory** | Rolling context window so the assistant remembers what you said |
| ⏱️ **Latency Profiling** | Per-stage timing breakdown printed after every turn |
| 🎛️ **CLI Overrides** | `--voice` and `--model` flags to customize on the fly |
| 🚫 **Zero FFmpeg** | Uses raw PCM playback — no external audio tools needed |

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **OpenAI API key** (with Whisper, Chat Completions, and TTS access)
- **Microphone + speakers** connected to your system

### 2. Install

```bash
# Clone & enter the project
cd voice_assistant

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-real-key-here
```

### 4. Run

```bash
python main.py
```

That's it. **Start talking.** 🎤

---

## 🎛️ CLI Options

```bash
# Pick a voice
python main.py --voice shimmer

# Pick a model
python main.py --model gpt-4o

# Combine both
python main.py --voice nova --model gpt-4o
```

**Available voices:** `alloy` · `echo` · `fable` · `onyx` · `nova` · `shimmer`

---

## 🧪 Testing

Test each pipeline stage independently:

```bash
python test_pipeline.py capture      # 🎤 Record 3s → print array shape
python test_pipeline.py transcribe   # 📝 Record → Whisper STT
python test_pipeline.py brain        # 🧠 Text → GPT streaming
python test_pipeline.py speak        # 🔊 Synthesize → play audio
python test_pipeline.py all          # 🔁 Run all tests in sequence
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `TTS_VOICE` | `alloy` | TTS voice name |
| `GPT_MODEL` | `gpt-4o-mini` | Chat model to use |
| `DEBUG` | `false` | Print debug info (VAD calibration, energy levels, etc.) |

---

## 📁 Project Structure

```
voice_assistant/
├── main.py              # 🎯 Entry point & conversation loop
├── capture.py           # 🎤 Mic recording + energy-based VAD
├── transcribe.py        # 📝 Whisper STT integration
├── brain.py             # 🧠 GPT chat + sentence chunker
├── speak.py             # 🔊 TTS synthesis + PCM playback
├── config.py            # ⚙️ Environment / config loader
├── test_pipeline.py     # 🧪 Per-stage test runner
├── requirements.txt     # 📦 Pinned dependencies
├── .env                 # 🔑 Your API key (git-ignored)
└── README.md            # 📖 You are here
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Audio Capture | `sounddevice` + `numpy` (16 kHz, mono, int16) |
| Voice Detection | Pure-Python energy-based VAD with auto-calibration |
| Speech-to-Text | OpenAI Whisper API |
| AI Brain | OpenAI GPT-4o / GPT-4o-mini (streaming) |
| Text-to-Speech | OpenAI TTS API (raw PCM output) |
| Audio Playback | `sounddevice` (24 kHz, no FFmpeg needed) |

---

<p align="center">
  Made with ❤️ , ☕ and a microphone<br/>
  <strong>Just speak. The assistant handles the rest.</strong>
</p>
