"""
Conversation context manager.

Maintains turn history and builds LLM prompts. Handles context
window truncation to keep the prompt within the model's limits.

Usage:
    conv = Conversation()
    conv.add_user_turn("Hi, I have a question about my booking")
    conv.add_assistant_turn("Sure, what's your question?")
    history = conv.get_history()  # -> list of ChatMessage
"""

from dataclasses import dataclass, field

from pipeline.llm import ChatMessage


# Rough estimate: 1 token ≈ 4 chars for English text
_CHARS_PER_TOKEN = 4
# Reserve tokens for system prompt + current user turn + LLM response
_RESERVED_TOKENS = 1024
# Llama 3.1 8B context window
_MAX_CONTEXT_TOKENS = 8192


class Conversation:
    """
    Manages conversation turn history for the LLM.
    """

    def __init__(self, max_context_tokens: int = _MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens
        self.turns: list[ChatMessage] = []

    def add_user_turn(self, text: str):
        """Record what the user said."""
        self.turns.append(ChatMessage(role="user", content=text))

    def add_assistant_turn(self, text: str):
        """Record what the assistant said."""
        self.turns.append(ChatMessage(role="assistant", content=text))

    def get_history(self) -> list[ChatMessage]:
        """
        Get conversation history, truncated to fit context window.

        Drops oldest turns first to stay within token budget.
        Always keeps at least the most recent exchange.
        """
        available_tokens = self.max_context_tokens - _RESERVED_TOKENS

        # Walk backwards, accumulating turns until we hit the limit
        kept: list[ChatMessage] = []
        token_count = 0

        for turn in reversed(self.turns):
            turn_tokens = len(turn.content) / _CHARS_PER_TOKEN
            if token_count + turn_tokens > available_tokens and len(kept) >= 2:
                break
            kept.append(turn)
            token_count += turn_tokens

        kept.reverse()
        return kept

    def get_last_user_text(self) -> str | None:
        """Get the most recent user message."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_turn_count(self) -> int:
        return len(self.turns)

    def clear(self):
        self.turns.clear()
