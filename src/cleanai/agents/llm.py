"""LLM factory and a small retry wrapper for Groq rate limits."""

from __future__ import annotations

import time

from langchain_groq import ChatGroq

from cleanai.config import get_settings


def build_llm(json_mode: bool = False, api_key: str | None = None) -> ChatGroq:
    settings = get_settings()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return ChatGroq(
        temperature=settings.llm_temperature,
        model_name=settings.groq_model_name,
        groq_api_key=api_key or settings.groq_api_key,
        model_kwargs=kwargs,
    )


def invoke_with_retry(llm, messages, attempts: int = 3):
    """Groq's free tier throttles aggressively; back off and retry."""
    last_error = None
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_error = exc
            if "rate" not in str(exc).lower() and "429" not in str(exc):
                raise
            time.sleep(2 ** (attempt + 1))
    raise last_error
