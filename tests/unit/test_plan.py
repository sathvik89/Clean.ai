from cleanai.core.plan import PlanStep, parse_plan, render_progress, render_step

VALID_TOOLS = ["handle_missing", "remove_duplicates", "convert_types"]
VALID_COLUMNS = ["age", "email"]


def test_parse_plain_json():
    raw = '{"steps": [{"id": "s1", "tool": "handle_missing", "columns": ["age"], "params": {"strategy": "median"}, "rationale": "fill nulls", "priority": 1}]}'
    steps = parse_plan(raw, VALID_TOOLS, VALID_COLUMNS)
    assert len(steps) == 1
    assert steps[0].tool == "handle_missing"
    assert steps[0].params == {"strategy": "median"}


def test_parse_handles_code_fences_and_noise():
    raw = 'Sure! Here is the plan:\n```json\n{"steps": [{"tool": "remove_duplicates"}]}\n```'
    steps = parse_plan(raw, VALID_TOOLS, VALID_COLUMNS)
    assert len(steps) == 1


def test_unknown_tools_and_columns_dropped():
    raw = '{"steps": [{"tool": "summon_demon", "columns": ["age"]}, {"tool": "convert_types", "columns": ["age", "ghost_col"]}]}'
    steps = parse_plan(raw, VALID_TOOLS, VALID_COLUMNS)
    assert len(steps) == 1
    assert steps[0].columns == ["age"]


def test_priority_sort_and_reindex():
    raw = ('{"steps": [{"tool": "handle_missing", "priority": 3},'
           '{"tool": "remove_duplicates", "priority": 1}]}')
    steps = parse_plan(raw, VALID_TOOLS, VALID_COLUMNS)
    assert steps[0].tool == "remove_duplicates"
    assert [s.id for s in steps] == ["s1", "s2"]


def test_garbage_returns_empty():
    assert parse_plan("the data looks great!", VALID_TOOLS, VALID_COLUMNS) == []


def test_render_progress_marks_current():
    steps = [PlanStep(id="s1", tool="a", status="done").model_dump(),
             PlanStep(id="s2", tool="b").model_dump()]
    text = render_progress(steps, 1)
    assert "[done]" in text
    assert "<- CURRENT" in text


def test_render_step_mentions_retry():
    step = PlanStep(id="s1", tool="handle_missing", attempts=1).model_dump()
    assert "previous attempts failed" in render_step(step)
