"""Tests for ContentChatbot."""

from unittest.mock import MagicMock, patch

import pytest

from src.analysis.chatbot import ContentChatbot


SAMPLE = (
    "The Acme Corp was founded in 1999. "
    "It sells widgets and gadgets worldwide."
)


def test_not_configured_raises():
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
        bot = ContentChatbot(SAMPLE)
        with pytest.raises(RuntimeError, match="not configured"):
            bot.answer("When was it founded?", [])


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
@patch("src.analysis.chatbot.OpenAI")
def test_answer_uses_context(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Founded in 1999."))]
    )

    bot = ContentChatbot(SAMPLE)
    out = bot.answer("When was Acme founded?", [])
    assert "1999" in out
    mock_client.chat.completions.create.assert_called_once()
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "CONTENT START" in system
    assert "CONTENT END" in system
    assert SAMPLE in system
    assert "scraped" in system.lower()
    assert "Page title:" not in system


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
@patch("src.analysis.chatbot.OpenAI")
def test_system_includes_page_title(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )

    bot = ContentChatbot(SAMPLE, page_title="About Acme")
    bot.answer("What is this page?", [])
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system = messages[0]["content"]
    assert "Page title: About Acme" in system
    assert "CONTENT START" in system


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
@patch("src.analysis.chatbot.OpenAI")
def test_long_context_truncated(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )

    long_text = "a" * 12_001
    bot = ContentChatbot(long_text)
    bot.answer("hi", [])
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system = messages[0]["content"]
    inner = system.split("CONTENT START", 1)[1].split("CONTENT END", 1)[0]
    assert "[Content truncated for length]" in inner
    assert inner.index("[Content truncated for length]") > 12_000


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
@patch("src.analysis.chatbot.OpenAI")
def test_empty_question(mock_openai_cls):
    bot = ContentChatbot(SAMPLE)
    with pytest.raises(ValueError):
        bot.answer(" ", [])


def test_is_configured():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "x"}, clear=False):
        assert ContentChatbot.is_configured() is True
