"""
config.py — Configuration loader for the Voice Assistant.

Loads environment variables from a .env file and exposes:
  - OPENAI_API_KEY  (required)
  - TTS_VOICE       (default: "alloy")
  - GPT_MODEL       (default: "gpt-4o-mini")
  - DEBUG           (default: False)
"""

import os
import sys
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Required ────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print(
        "\n❌  OPENAI_API_KEY is not set.\n"
        "   1. Copy .env.example → .env\n"
        "   2. Paste your OpenAI API key into the OPENAI_API_KEY field.\n"
    )
    sys.exit(1)

# ── Optional (with defaults) ────────────────────────────────────────
TTS_VOICE: str = os.getenv("TTS_VOICE", "alloy")
GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
