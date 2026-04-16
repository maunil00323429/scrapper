"""
Q&A over scraped page text using OpenAI chat completions.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 12_000
_TRUNCATION_NOTE = "[Content truncated for length]"
_DEFAULT_MODEL = "gpt-3.5-turbo"

_SYSTEM_INTRO = (
    "You are a helpful assistant answering questions about a web page that the user "
    "has scraped and is analyzing. The full extracted text from that page is your only "
    "source of facts; it appears in the delimited block below. When the user asks about "
    "\"the article\", \"this page\", \"the content\", \"the scraped data\", \"the text\", "
    "or similar wording, they mean that extracted text. Answer every question using "
    "only that content. If something cannot be answered from the content, say so "
    "briefly and mention what the page appears to be about based on what is available. "
    "Keep answers concise and directly relevant to the question."
)


class ContentChatbot:
    """
    Answer user questions using page text as the only knowledge source.
    """

    def __init__(
        self,
        context_text: str,
        *,
        page_title: str | None = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        raw = (context_text or "").strip()
        truncated = len(raw) > _MAX_CONTEXT_CHARS
        body = raw[:_MAX_CONTEXT_CHARS] if truncated else raw
        if truncated:
            body = f"{body.rstrip()}\n\n{_TRUNCATION_NOTE}"
        self._context_body = body
        self._page_title = (page_title or "").strip() or None
        self._model = model

    def _system_message(self) -> str:
        parts: list[str] = [_SYSTEM_INTRO, ""]
        if self._page_title:
            parts.append(f"Page title: {self._page_title}")
            parts.append("")
        parts.append("CONTENT START")
        parts.append(self._context_body)
        parts.append("CONTENT END")
        return "\n".join(parts)

    @staticmethod
    def is_configured() -> bool:
        return bool((os.environ.get("OPENAI_API_KEY") or "").strip())

    def answer(self, question: str, conversation_history: list[dict]) -> str:
        if not self.is_configured() or OpenAI is None:
            raise RuntimeError(
                "OpenAI is not configured. Set OPENAI_API_KEY in your environment."
            )
        q = (question or "").strip()
        if not q:
            raise ValueError("Question must not be empty")

        system = self._system_message()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": q})

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
            )
            out = completion.choices[0].message.content
            return (out or "").strip() or "(No response)"
        except Exception as exc:
            logger.warning("Chat completion failed: %s", exc, exc_info=True)
            raise RuntimeError(str(exc)) from exc
