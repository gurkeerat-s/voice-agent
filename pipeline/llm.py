"""
Async LLM client for vLLM with streaming and prefix caching.

vLLM runs as a separate process serving an OpenAI-compatible API.
This client streams tokens back and detects sentence boundaries
so TTS can start generating audio on the first complete sentence.

Usage:
    llm = LLMClient()

    # Warm KV cache with partial transcript (while user still speaking)
    await llm.warm_prefix(partial_text, history)

    # Stream full response sentence-by-sentence
    async for sentence in llm.generate_stream(final_text, history):
        # send each sentence to TTS immediately
        tts.synthesize(sentence)
"""

import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from config import config


# Regex to split on sentence-ending punctuation followed by space or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])$')


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


class LLMClient:
    """
    Async streaming client for vLLM's OpenAI-compatible API.
    """

    def __init__(self):
        cfg = config.llm
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.system_prompt = cfg.system_prompt

        self.client = AsyncOpenAI(
            api_key="not-needed",  # vLLM doesn't require a real key
            base_url=cfg.api_base,
        )

    def _build_messages(
        self, user_text: str, history: list[ChatMessage]
    ) -> list[dict]:
        """Build the messages array for the OpenAI chat API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_text})
        return messages

    async def warm_prefix(
        self, partial_text: str, history: list[ChatMessage]
    ) -> None:
        """
        Send a partial transcript to vLLM to warm the KV cache.

        With prefix caching enabled, vLLM caches the KV states for the
        system prompt + history + partial transcript. When the full
        transcript arrives, only the new tokens need processing.

        This is fire-and-forget — we don't use the response.
        """
        if not partial_text.strip():
            return

        messages = self._build_messages(partial_text, history)

        try:
            # Request with max_tokens=1 just to warm the cache
            await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,
                stream=False,
            )
        except Exception:
            # Cache warming is best-effort — don't crash on failure
            pass

    async def generate_stream(
        self, user_text: str, history: list[ChatMessage]
    ):
        """
        Stream the LLM response, yielding complete sentences.

        Each yielded string is a complete sentence (ends with . ! or ?),
        ready to be sent to TTS. The final yield may not end with
        punctuation if the LLM stops mid-sentence.

        Yields:
            str: A complete sentence or the final fragment.
        """
        messages = self._build_messages(user_text, history)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        buffer = ""
        full_response = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            buffer += delta
            full_response += delta

            # Try to split on sentence boundaries
            sentences = _SENTENCE_END.split(buffer)

            if len(sentences) > 1:
                # All but the last are complete sentences
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence

                # Keep the incomplete remainder in the buffer
                buffer = sentences[-1]

        # Yield whatever's left in the buffer
        if buffer.strip():
            yield buffer.strip()

    async def generate_full(
        self, user_text: str, history: list[ChatMessage]
    ) -> str:
        """
        Non-streaming generation. Returns the full response as one string.
        Useful for testing.
        """
        messages = self._build_messages(user_text, history)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )

        return response.choices[0].message.content or ""
