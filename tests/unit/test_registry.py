import pandas as pd

from cleanai.agents.tools import registry


def test_unknown_tool_returns_error(messy_df):
    result = registry.execute("teleport_data", messy_df, {})
    assert result.status == "error"
    assert "unknown tool" in result.message


def test_bad_arguments_are_rejected_not_raised(messy_df):
    result = registry.execute("clean_text", messy_df, {"columns": "not-a-list"})
    assert result.status == "error"
    assert "bad arguments" in result.message


def test_legacy_tool_names_are_normalized():
    assert registry.normalize_tool_name("tool_handle_missing") == "handle_missing"
    assert registry.normalize_tool_name("execute_custom_pandas") == "run_pandas"
    assert registry.normalize_tool_name("handle_categorical") == "encode_categorical"


def test_mutating_tool_returns_new_df(messy_df):
    result = registry.execute("remove_duplicates", messy_df, {})
    assert result.mutated
    assert len(result.df) < len(messy_df)
    assert len(messy_df) == 5  # input untouched


def test_readonly_tool_returns_message_only(messy_df):
    result = registry.execute("get_summary", messy_df, {})
    assert not result.mutated
    assert "dataset:" in result.message


def test_report_blocked_sets_status(messy_df):
    result = registry.execute("report_blocked", messy_df, {"reason": "column is unfixable"})
    assert result.status == "blocked"


def test_handler_failures_are_captured(messy_df):
    result = registry.execute("run_pandas", messy_df, {"code": "df = df[df['nope']]"})
    assert result.status == "error"


def test_langchain_tools_match_registry():
    tools = registry.langchain_tools()
    assert {t.name for t in tools} == set(registry.REGISTRY.keys())
