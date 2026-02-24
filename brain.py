"""
brain.py — Stage 3: AI Brain.

Manages conversation history and streams responses from the OpenAI
Chat Completions API, yielding complete sentences for TTS handoff.
"""

import re
from typing import Generator
from openai import OpenAI
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. "
    "Keep every response short, conversational, and under 3 sentences. "
    "Do not use markdown, bullet points, or numbered lists — "
    "your answers will be spoken aloud."
)


class ConversationBuffer:
    """Rolling list of chat messages in OpenAI format."""

    def __init__(self, max_turns: int = 20):
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def get_messages(self) -> list[dict[str, str]]:
        """Return messages with the system prompt prepended."""
        return [{"role": "system", "content": SYSTEM_PROMPT}] + list(
            self._messages
        )

    def _trim(self) -> None:
        """Keep only the most recent max_turns * 2 messages."""
        limit = self._max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]


def generate_response_stream(
    buffer: ConversationBuffer,
) -> Generator[str, None, None]:
    """
    Call the Chat Completions API with streaming and yield text chunks.

    After the full response has been streamed, the complete text is
    automatically added to the conversation buffer.

    Yields
    ------
    str
        Incremental text chunks as they arrive from the API.
    """
    full_response_parts: list[str] = []

    try:
        stream = _client.chat.completions.create(
            model=config.GPT_MODEL,
            messages=buffer.get_messages(),
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response_parts.append(delta.content)
                yield delta.content

    except Exception as exc:
        error_msg = "Sorry, I had trouble thinking of a response."
        if config.DEBUG:
            print(f"  ⚠️  Chat API error: {exc}")
        full_response_parts.append(error_msg)
        yield error_msg

    finally:
        full_response = "".join(full_response_parts)
        if full_response:
            buffer.add_assistant_message(full_response)


# ── Sentence Chunker ────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentence_chunker(
    stream: Generator[str, None, None],
) -> Generator[str, None, None]:
    """
    Consume a text stream and yield complete sentences.

    Splits on sentence-ending punctuation (. ? !) followed by whitespace.
    Any remaining text is flushed at the end of the stream.

    Yields
    ------
    str
        One complete sentence at a time.
    """
    buffer = ""

    for chunk in stream:
        buffer += chunk

        # Try to split off complete sentences
        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            sentence = buffer[: match.start() + 1].strip()
            buffer = buffer[match.end() :]
            if sentence:
                yield sentence

    # Flush remaining text
    remaining = buffer.strip()
    if remaining:
        yield remaining
