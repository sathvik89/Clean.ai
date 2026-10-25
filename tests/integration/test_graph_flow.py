"""Runs the whole analyzer -> planner -> executor -> validator graph with a
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


PLAN_JSON = json.dumps({"steps": [
    {"id": "s1", "tool": "convert_types", "columns": [],
     "params": {"column_type_map": {"age": "float"}},
     "rationale": "age is numeric stored as text", "priority": 1},
    {"id": "s2", "tool": "remove_duplicates", "columns": [], "params": {},
     "rationale": "drop exact duplicate rows", "priority": 1},
    {"id": "s3", "tool": "handle_missing", "columns": ["age"],
     "params": {"strategy": "median"}, "rationale": "fill missing age", "priority": 2},
]})


def tool_call_message(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted_run_messages():
    return [
        AIMessage(content="analysis: age stored as text, duplicates present"),
        AIMessage(content=PLAN_JSON),
        tool_call_message("convert_types", {"column_type_map": {"age": "float"}}, "c1"),
        tool_call_message("remove_duplicates", {}, "c2"),
        tool_call_message("handle_missing", {"strategy": "median", "columns": ["age"]}, "c3"),
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

    # plan steps were tracked to completion
    assert all(step["status"] == "done" for step in final["plan_steps"])
    # ledger recorded every action with its effect
    assert len(final["ledger"]) == 3
    assert all(record["status"] == "ok" for record in final["ledger"])


def test_duplicate_calls_are_short_circuited(monkeypatch, isolated_storage):
    from cleanai.agents import nodes
    from cleanai.agents.graph import build_graph
    from cleanai.agents.state import initial_state

    clean_df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 1000.0]})
    messages = [
        AIMessage(content="analysis"),
        AIMessage(content=json.dumps({"steps": [
            {"id": "s1", "tool": "detect_outliers", "columns": ["value"], "params": {},
             "rationale": "inspect", "priority": 1},
        ]})),
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
    messages = [
        AIMessage(content="analysis"),
        AIMessage(content=json.dumps({"steps": [
            {"id": "s1", "tool": "handle_missing", "columns": ["v"],
             "params": {"strategy": "drop"}, "rationale": "drop nulls", "priority": 1},
        ]})),
        tool_call_message("handle_missing", {"strategy": "drop", "columns": ["v"]}, "c1"),
        tool_call_message("handle_missing", {"strategy": "median", "columns": ["v"]}, "c2"),
    ]
    llm = make_scripted_llm(messages)
    monkeypatch.setattr(nodes, "build_llm", lambda *a, **k: llm)
    graph = build_graph(with_human_review=False)

    config = {"configurable": {"thread_id": "guard-run"}}
    for _ in graph.stream(initial_state(df, "g.csv", "clean"), config):
        pass
    final = graph.get_state(config).values

    # the drop (80% of rows) was refused, the median fill then succeeded
    assert len(final["current_df"]) == 5
    assert final["current_df"]["v"].isnull().sum() == 0
    assert any(record["status"] == "error" and "guardrail" in record["error"]
               for record in final["ledger"])
