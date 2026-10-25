import pandas as pd
import pytest

from cleanai.services.auth import AuthError, UserStore
from cleanai.services.persistence import RunStore
from cleanai.services.pipeline import simulate_plan
from cleanai.services.recipe import build_recipe_script
from cleanai.services.reporting import build_run_report


@pytest.fixture
def summary():
    return {
        "dataset_name": "orders.csv", "user_context": "ML modeling",
        "status": "clean", "error": "",
        "rows_before": 100, "rows_after": 95, "cols_before": 5, "cols_after": 5,
        "nulls_before": 12, "nulls_after": 0,
        "duplicates_before": 5, "duplicates_after": 0,
        "score_before": 55, "score_after": 97, "tokens_used": 4200,
        "steps": [{"id": "s1", "tool": "remove_duplicates", "columns": [],
                   "status": "done", "result": "rows 100->95"}],
        "ledger": [{"seq": 1, "tool": "remove_duplicates", "args": {},
                    "status": "ok", "delta": "rows 100->95", "error": ""}],
        "logs_text": "done", "analysis_report": "all good",
        "cleaned_csv": "a,b\n1,2\n",
    }


def test_register_and_login(tmp_path):
    store = UserStore(tmp_path / "users.sqlite3")
    user = store.register("sam@example.com", "supersecret1")
    assert user["email"] == "sam@example.com"
    assert store.authenticate("SAM@example.com", "supersecret1")["id"] == user["id"]
    assert store.authenticate("sam@example.com", "wrongpass11") is None


def test_register_rejects_bad_input(tmp_path):
    store = UserStore(tmp_path / "users.sqlite3")
    with pytest.raises(AuthError, match="valid email"):
        store.register("not-an-email", "supersecret1")
    with pytest.raises(AuthError, match="at least 8"):
        store.register("ok@example.com", "short")
    store.register("dup@example.com", "supersecret1")
    with pytest.raises(AuthError, match="already registered"):
        store.register("dup@example.com", "supersecret1")


def test_run_store_roundtrip(tmp_path, summary):
    store = RunStore(tmp_path / "runs.sqlite3")
    run_id = store.save_run(user_id=7, payload=summary)
    runs = store.list_runs(user_id=7)
    assert len(runs) == 1
    assert runs[0]["n_steps"] == 1
    fetched = store.get_run(run_id, user_id=7)
    assert fetched["cleaned_csv"] == "a,b\n1,2\n"
    assert store.get_run(run_id, user_id=999) is None


def test_simulate_plan_reports_effects(messy_df):
    impacts = simulate_plan(messy_df, [
        {"id": "s1", "tool": "remove_duplicates", "columns": ["email"], "params": {}},
    ])
    assert impacts[0]["ok"]
    assert "rows" in impacts[0]["effect"]


def test_recipe_script_is_valid_python(summary):
    script = build_recipe_script(summary)
    compile(script, "recipe.py", "exec")
    assert "drop_duplicates" in script


def test_run_report_contains_metrics(summary):
    report = build_run_report(summary)
    assert "quality score | 55 | 97" in report
    assert "remove_duplicates" in report


def test_apply_recorded_actions_replays_deterministically(messy_df):
    from cleanai.services.pipeline import apply_recorded_actions

    actions = [
        {"seq": 1, "tool": "remove_duplicates", "args": {}, "status": "ok", "delta": "rows 5->4"},
        {"seq": 2, "tool": "handle_missing", "args": {"strategy": "median", "columns": ["age"]},
         "status": "ok", "delta": "nulls.age 1->0"},
        {"seq": 3, "tool": "get_summary", "args": {}, "status": "ok", "delta": ""},
    ]
    replay = apply_recorded_actions(messy_df, actions)
    assert len(replay["df"]) == 4
    assert replay["df"]["age"].isnull().sum() == 0
    assert replay["score_after"] >= replay["score_before"]
    assert len(replay["log"]) == 2


def test_tool_usage_counts_effective_calls(tmp_path, summary):
    from cleanai.services.persistence import RunStore

    store = RunStore(tmp_path / "runs.sqlite3")
    store.save_run(user_id=1, payload=summary)
    usage = store.tool_usage(user_id=1)
    assert usage == {"remove_duplicates": 1}
