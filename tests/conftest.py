import os
from collections import deque

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedLLM(GenericFakeChatModel):
    """Fake chat model that replays a fixed sequence of AIMessages and
    tolerates bind_tools, so the whole graph runs without any API."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_scripted_llm(messages) -> ScriptedLLM:
    return ScriptedLLM(messages=iter(list(messages)))


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Point all sqlite stores at a temp dir and reset the settings cache."""
    from cleanai import config

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))
    config.get_settings.cache_clear()
    yield config.get_settings()
    config.get_settings.cache_clear()


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 2, 4, 5],
        "age": ["25", "199", "199", None, "31"],
        "email": ["a@x.com", "b@x.com", "b@x.com", "not-an-email", "c@x.com"],
        "name": ["  Alice ", "bob", "bob", "Dana  ", "Eve"],
    })
