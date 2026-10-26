"""Runs the whole analyzer -> eda -> executor -> validator graph with a
scripted fake LLM, so the agent loop is tested deterministically and
without any API key."""

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from tests.conftest import make_scripted_llm


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "age": ["25", "30", "30", None],
        "city": ["pune", "delhi", "delhi", "goa"],
    })


EDA_REPORT_JSON = json.dumps({
    "narrative": {
        "overview": "Small table with age stored as text and one duplicate row.",
        "key_findings": ["age is numeric text", "one exact duplicate"],
        "risks": ["missing age value"],
    },
    "suggestions": [
        {"title": "Convert age to numbers", "why": "stored as text", "severity": "high",
         "tool": "convert_types", "columns": [], "params": {"column_type_map": {"age": "float"}}},
        {"title": "Remove duplicate rows", "why": "one exact duplicate", "severity": "high",
         "tool": "remove_duplicates", "columns": [], "params": {}},
        {"title": "Fill missing age", "why": "one null", "severity": "medium",
         "tool": "handle_missing", "columns": ["age"], "params": {"strategy": "median"}},
    ],
})


def tool_call_message(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted_run_messages():
    # every plan step is fully specified, so execution is mechanical:
    # the model is only consulted during eda
    return [
        # eda investigation: one lookup, then a closing note
        tool_call_message("value_counts", {"column": "age"}, "i1"),
        AIMessage(content="age holds numeric text; city labels look fine."),
        # eda report with suggestions
        AIMessage(content=EDA_REPORT_JSON),
    ]


@pytest.fixture
def scripted_graph(monkeypatch, isolated_storage):
    from cleanai.agents import nodes
    from cleanai.agents.graph import build_graph

    llm = make_scripted_llm(scripted_run_messages())
    monkeypatch.setattr(nodes, "build_llm", lambda *a, **k: llm)
    return build_graph(with_human_review=False)


def test_full_run_cleans_the_data(scripted_graph, dirty_df):
    from cleanai.agents.state import initial_state

    config = {"configurable": {"thread_id": "test-run"}}
    state = initial_state(dirty_df, "dirty.csv", "ML modeling")
    for _ in scripted_graph.stream(state, config):
        pass
    final = scripted_graph.get_state(config).values

    assert final["is_clean"]
    assert final["run_error"] == ""
    df = final["current_df"]
    assert len(df) == 3                      # duplicate row dropped
    assert df["age"].isnull().sum() == 0     # nulls filled
    assert str(df["age"].dtype) == "float64"

    # eda produced a narrative and suggestions, which compiled into the plan
    assert "duplicate" in final["analysis_report"]
    assert len(final["suggestions"]) == 3
    assert len(final["plan_steps"]) == 4   # 3 suggestions + safety-net fill
    # structure fix ordered before value fixes regardless of model order
    assert final["plan_steps"][0]["tool"] == "convert_types"
    assert all(step["status"] == "done" for step in final["plan_steps"])
    assert len(final["ledger"]) == 4
    assert all(record["status"] in ("ok", "noop") for record in final["ledger"])


def test_duplicate_calls_are_short_circuited(monkeypatch, isolated_storage):
    from cleanai.agents import nodes
    from cleanai.agents.graph import build_graph
    from cleanai.agents.state import initial_state

    clean_df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 1000.0]})
    # flag_rows without params is underspecified, so the LLM executor handles it
    report = json.dumps({"narrative": {"overview": "fine"}, "suggestions": [
        {"title": "Flag the outliers", "severity": "low",
         "tool": "flag_rows", "columns": ["value"], "params": {}},
    ]})
    messages = [
        AIMessage(content="nothing surprising"),          # eda: no lookups
        AIMessage(content=report),
        # same read-only call twice: second one must be blocked as duplicate
        tool_call_message("detect_outliers", {"columns": ["value"]}, "c1"),
        tool_call_message("detect_outliers", {"columns": ["value"]}, "c2"),
        tool_call_message("report_blocked", {"reason": "nothing else to do"}, "c3"),
    ]
    llm = make_scripted_llm(messages)
    monkeypatch.setattr(nodes, "build_llm", lambda *a, **k: llm)
    graph = build_graph(with_human_review=False)

    config = {"configurable": {"thread_id": "dup-run"}}
    for _ in graph.stream(initial_state(clean_df, "d.csv", "check"), config):
        pass
    final = graph.get_state(config).values
    logs = "\n".join(final["logs"])
    assert "skipped duplicate call" in logs


def test_row_drop_guardrail_blocks_mass_deletion(monkeypatch, isolated_storage):
    from cleanai.agents import nodes
    from cleanai.agents.graph import build_graph
    from cleanai.agents.state import initial_state

    df = pd.DataFrame({"v": [1.0, None, None, None, 5.0], "k": [1, 2, 3, 4, 5]})
    report = json.dumps({"narrative": {"overview": "nulls in v"}, "suggestions": [
        {"title": "Drop null rows", "severity": "high",
         "tool": "handle_missing", "columns": ["v"], "params": {"strategy": "drop"}},
    ]})
    messages = [
        AIMessage(content="v is mostly missing"),
        AIMessage(content=report),
        # the drop step runs mechanically and is refused; the model only
        # appears in the repair phase, where it fills instead
        tool_call_message("handle_missing", {"strategy": "median", "columns": ["v"]}, "c2"),
    ]
    llm = make_scripted_llm(messages)
    monkeypatch.setattr(nodes, "build_llm", lambda *a, **k: llm)
    graph = build_graph(with_human_review=False)

    config = {"configurable": {"thread_id": "guard-run"}}
    for _ in graph.stream(initial_state(df, "g.csv", "clean"), config):
        pass
    final = graph.get_state(config).values

    # the drop (60% of rows) was refused, the median fill then succeeded
    assert len(final["current_df"]) == 5
    assert final["current_df"]["v"].isnull().sum() == 0
    assert any(record["status"] == "error" and "guardrail" in record["error"]
               for record in final["ledger"])


def test_unusable_report_falls_back_to_deterministic_suggestions(monkeypatch, isolated_storage, dirty_df):
    from cleanai.agents import nodes
    from cleanai.agents.graph import build_graph
    from cleanai.agents.state import initial_state

    messages = [
        AIMessage(content="looks odd"),
        AIMessage(content="sorry, I cannot produce JSON today"),
        # fallback suggestions compile into a fully-specified plan that
        # executes mechanically - no further model calls needed
    ]
    llm = make_scripted_llm(messages)
    monkeypatch.setattr(nodes, "build_llm", lambda *a, **k: llm)
    graph = build_graph(with_human_review=False)

    config = {"configurable": {"thread_id": "fallback-run"}}
    for _ in graph.stream(initial_state(dirty_df, "f.csv", ""), config):
        pass
    final = graph.get_state(config).values
    assert final["suggestions"], "fallback suggestions should exist"
    assert final["plan_steps"], "plan should compile from fallback suggestions"
