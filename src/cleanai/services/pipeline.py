"""High-level run orchestration shared by the UI and the CLI.

A CleaningSession owns one run: it streams the graph up to the human
review interrupt, lets the caller approve or edit the plan, resumes to
completion, and finally persists the run and feeds the case memory.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from cleanai.agents.graph import build_graph
from cleanai.agents.state import initial_state
from cleanai.config import get_settings
from cleanai.core import profiling
from cleanai.core.memory import CaseMemory, dataset_fingerprint

# which issue kinds each tool plausibly addresses, used to turn a finished
# run into case-memory strategies
TOOL_ISSUE_MAP = {
    "handle_missing": {"missing_values"},
    "remove_duplicates": {"duplicate_rows", "duplicate_key_values"},
    "convert_types": {"wrong_dtype", "non_numeric_values", "invalid_dates", "mixed_types"},
    "fix_outliers": {"out_of_range", "negative_values"},
    "clean_text": {"mixed_types"},
    "run_pandas": {"out_of_range", "negative_values", "invalid_email", "future_dates"},
}

ProgressCallback = Callable[[str, Dict[str, Any]], None]


class CleaningSession:
    def __init__(self, df: pd.DataFrame, file_name: str, user_context: str):
        self.app = build_graph()
        self.thread_id = uuid.uuid4().hex[:12]
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self._start_state = initial_state(df, file_name, user_context)
        self.file_name = file_name
        self.user_context = user_context
        self._before_snapshot = profiling.snapshot(df)
        self._initial_issue_kinds = {
            issue.kind for issue in profiling.all_open_issues(df, profiling.build_cards(df))
        }
        self._score_before = profiling.quality_score(df)

    # ---- phase 1: analyze + plan, stop at human review ----
    def run_until_review(self, on_event: Optional[ProgressCallback] = None) -> None:
        for event in self.app.stream(self._start_state, self.config):
            for node_name, output in event.items():
                if on_event:
                    on_event(node_name, output or {})

    # ---- plan inspection / editing ----
    def state_values(self) -> Dict[str, Any]:
        return self.app.get_state(self.config).values

    def awaiting_review(self) -> bool:
        return bool(self.app.get_state(self.config).next)

    def get_plan(self) -> List[Dict[str, Any]]:
        return list(self.state_values().get("plan_steps", []))

    def update_plan(self, steps: List[Dict[str, Any]]) -> None:
        self.app.update_state(self.config, {"plan_steps": steps, "current_step_index": 0})

    # ---- phase 2: execute to completion ----
    def resume(self, on_event: Optional[ProgressCallback] = None) -> None:
        for event in self.app.stream(None, self.config):
            for node_name, output in event.items():
                if on_event:
                    on_event(node_name, output or {})

    # ---- results ----
    def result_summary(self) -> Dict[str, Any]:
        values = self.state_values()
        df_after = values["current_df"]
        after = profiling.snapshot(df_after)
        before = self._before_snapshot
        status = "clean" if values.get("is_clean") else ("stopped" if values.get("run_error") else "partial")
        return {
            "dataset_name": self.file_name,
            "user_context": self.user_context,
            "status": status,
            "error": values.get("run_error", ""),
            "rows_before": before["rows"], "rows_after": after["rows"],
            "cols_before": before["cols"], "cols_after": after["cols"],
            "nulls_before": sum(before["nulls"].values()),
            "nulls_after": sum(after["nulls"].values()),
            "duplicates_before": before["duplicates"],
            "duplicates_after": after["duplicates"],
            "score_before": self._score_before,
            "score_after": profiling.quality_score(df_after),
            "steps": values.get("plan_steps", []),
            "ledger": values.get("ledger", []),
            "logs_text": "\n".join(values.get("logs", [])),
            "analysis_report": values.get("analysis_report", ""),
            "cleaned_csv": df_after.to_csv(index=False),
        }

    def learn_from_run(self) -> None:
        """Feed the case memory so future runs on similar data plan better."""
        values = self.state_values()
        df_after = values["current_df"]
        final_issue_kinds = {
            issue.kind
            for issue in profiling.all_open_issues(df_after, profiling.build_cards(df_after))
        }
        strategies = []
        for record in values.get("ledger", []):
            addressed = TOOL_ISSUE_MAP.get(record["tool"], set())
            for kind in addressed & self._initial_issue_kinds:
                strategies.append({
                    "issue_kind": kind,
                    "tool": record["tool"],
                    "succeeded": record["status"] == "ok" and kind not in final_issue_kinds,
                    "detail": record.get("delta", "")[:140],
                })
        if not strategies:
            return
        outcome = 1.0 if values.get("is_clean") else max(
            0.0, 1.0 - len(final_issue_kinds) / max(len(self._initial_issue_kinds), 1)
        )
        original_cards = profiling.build_cards(self.app.get_state(self.config).values["original_df"])
        CaseMemory(get_settings().cases_db_path).store_case(
            fingerprint=dataset_fingerprint(original_cards),
            outcome_score=outcome,
            strategies=strategies,
        )
