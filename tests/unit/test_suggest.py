import json

from cleanai.core import profiling, suggest

VALID_TOOLS = ["normalize_nulls", "convert_types", "remove_duplicates",
               "handle_missing", "clean_text", "encode_categorical"]
VALID_COLUMNS = ["age", "email", "name"]


def make_raw(items):
    return json.dumps({"suggestions": items})


def test_parse_valid_suggestions():
    raw = make_raw([{"title": "Fix age", "why": "text numbers", "severity": "high",
                     "tool": "convert_types", "columns": ["age"],
                     "params": {"column_type_map": {"age": "float"}}}])
    items = suggest.parse_suggestions(raw, VALID_TOOLS, VALID_COLUMNS)
    assert len(items) == 1
    assert items[0].tool == "convert_types"
    assert items[0].accepted


def test_unknown_tools_and_columns_dropped():
    raw = make_raw([
        {"title": "x", "tool": "summon_demon", "columns": ["age"]},
        {"title": "y", "tool": "clean_text", "columns": ["age", "ghost"]},
    ])
    items = suggest.parse_suggestions(raw, VALID_TOOLS, VALID_COLUMNS)
    assert len(items) == 1
    assert items[0].columns == ["age"]


def test_build_plan_orders_structure_before_values():
    suggestions = [
        {"id": "a1", "title": "fill", "tool": "handle_missing", "columns": ["age"],
         "params": {}, "severity": "high", "source": "agent", "accepted": True, "why": ""},
        {"id": "a2", "title": "nulls", "tool": "normalize_nulls", "columns": [],
         "params": {}, "severity": "medium", "source": "agent", "accepted": True, "why": ""},
        {"id": "a3", "title": "skip me", "tool": "remove_duplicates", "columns": [],
         "params": {}, "severity": "high", "source": "agent", "accepted": False, "why": ""},
    ]
    plan = suggest.build_plan(suggestions)
    # normalize_nulls reveals nulls, so a catch-all fill is appended too
    assert [s["tool"] for s in plan] == ["normalize_nulls", "handle_missing", "handle_missing"]
    assert [s["id"] for s in plan] == ["s1", "s2", "s3"]


def test_fallback_suggestions_cover_detected_issues(messy_df):
    cards = profiling.build_cards(messy_df)
    items = suggest.fallback_suggestions(messy_df, cards)
    tools = {s.tool for s in items}
    assert "remove_duplicates" in tools
    assert "handle_missing" in tools


def test_narrative_markdown_renders_sections():
    md = suggest.narrative_to_markdown({
        "overview": "A small messy table.",
        "key_findings": ["age is text"],
        "risks": ["nulls remain"],
    })
    assert "A small messy table." in md
    assert "Key findings" in md
    assert "- age is text" in md


def test_user_source_ids():
    raw = make_raw([{"title": "lower emails", "tool": "clean_text", "columns": ["email"]}])
    items = suggest.parse_suggestions(raw, VALID_TOOLS, VALID_COLUMNS, source="user")
    assert items[0].source == "user"
    assert items[0].id.startswith("u")


def test_build_plan_appends_catch_all_fill_after_null_revealing_steps():
    suggestions = [
        {"id": "a1", "title": "coerce", "tool": "convert_types", "columns": [],
         "params": {"column_type_map": {"age": "float"}}, "severity": "high",
         "source": "agent", "accepted": True, "why": ""},
    ]
    plan = suggest.build_plan(suggestions)
    assert plan[-1]["tool"] == "handle_missing"
    assert "safety net" in plan[-1]["rationale"]


def test_no_catch_all_when_fill_already_covers_everything():
    suggestions = [
        {"id": "a1", "title": "coerce", "tool": "convert_types", "columns": [],
         "params": {}, "severity": "high", "source": "agent", "accepted": True, "why": ""},
        {"id": "a2", "title": "fill all", "tool": "handle_missing", "columns": [],
         "params": {"strategy": "auto"}, "severity": "medium",
         "source": "agent", "accepted": True, "why": ""},
    ]
    plan = suggest.build_plan(suggestions)
    assert sum(1 for s in plan if s["tool"] == "handle_missing") == 1
