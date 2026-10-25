from cleanai.core import profiling
from cleanai.core.memory import (
    ActionRecord, CaseMemory, dataset_fingerprint, render_cards, render_ledger,
)


def make_record(seq, tool="handle_missing", status="ok", delta="nulls.age 3->0"):
    return ActionRecord(seq=seq, tool=tool, args={"strategy": "median"},
                        status=status, delta=delta).to_dict()


def test_ledger_renders_compact_lines():
    text = render_ledger([make_record(1), make_record(2, status="error", delta="")])
    assert "#1 handle_missing" in text
    assert "-> ok | nulls.age 3->0" in text


def test_ledger_collapses_old_entries():
    records = [make_record(i) for i in range(1, 25)]
    text = render_ledger(records, max_entries=10)
    assert "collapsed" in text
    assert "#24" in text
    assert "#5 " not in text


def test_fingerprint_is_deterministic(messy_df):
    cards = profiling.build_cards(messy_df)
    assert dataset_fingerprint(cards) == dataset_fingerprint(cards)
    assert any(t.startswith("issue:") for t in dataset_fingerprint(cards))


def test_case_memory_returns_hints_for_similar_data(tmp_path, messy_df):
    memory = CaseMemory(tmp_path / "cases.sqlite3")
    fingerprint = dataset_fingerprint(profiling.build_cards(messy_df))
    memory.store_case(fingerprint, outcome_score=1.0, strategies=[
        {"issue_kind": "missing_values", "tool": "handle_missing", "succeeded": True},
        {"issue_kind": "missing_values", "tool": "handle_missing", "succeeded": True},
    ])
    hints = memory.hints_for(fingerprint)
    assert "handle_missing succeeded" in hints


def test_case_memory_ignores_dissimilar_and_failed_runs(tmp_path, messy_df):
    memory = CaseMemory(tmp_path / "cases.sqlite3")
    fingerprint = dataset_fingerprint(profiling.build_cards(messy_df))
    memory.store_case(["role:text", "ncols:50+"], 1.0,
                      [{"issue_kind": "x", "tool": "y", "succeeded": True}])
    memory.store_case(fingerprint, 0.1,
                      [{"issue_kind": "missing_values", "tool": "handle_missing", "succeeded": True}])
    assert memory.hints_for(fingerprint) == ""


def test_render_cards_dirty_filter(messy_df):
    cards = profiling.build_cards(messy_df)
    for card in cards:
        card.issues_open = []
        card.status = "clean"
    assert render_cards(cards, only_dirty=True) == "all columns are clean"
