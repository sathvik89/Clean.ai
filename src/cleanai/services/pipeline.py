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
    "normalize_nulls": {"disguised_nulls"},
    "handle_missing": {"missing_values"},
    "remove_duplicates": {"duplicate_rows", "duplicate_key_values"},
    "convert_types": {"wrong_dtype", "non_numeric_values", "invalid_dates", "mixed_types"},
    "fix_outliers": {"out_of_range", "negative_values"},
    "clean_text": {"mixed_types"},
    "run_pandas": {"out_of_range", "negative_values", "invalid_email", "future_dates"},
    "extract_numbers": {"non_numeric_values", "wrong_dtype", "mixed_types"},
    "parse_dates": {"invalid_dates", "wrong_dtype", "future_dates"},
    "standardize_categories": {"mixed_types"},
    "impute_group": {"missing_values"},
    "replace_values": {"out_of_range", "negative_values", "non_numeric_values"},
}

ProgressCallback = Callable[[str, Dict[str, Any]], None]


def simulate_plan(df: pd.DataFrame, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dry-run every plan step on a copy and report what each would change.

    Shown in the review screen so approval is an informed decision.
    """
    from cleanai.agents.tools import registry

    working = df.copy()
    impacts = []
    for step in steps:
        args = dict(step.get("params") or {})
        if step.get("columns"):
            args.setdefault("columns", step["columns"])
        before = profiling.snapshot(working)
        result = registry.execute(step["tool"], working, args)
        impact = {"id": step.get("id"), "tool": step["tool"], "ok": result.status == "ok"}
        if result.mutated and result.status == "ok":
            after = profiling.snapshot(result.df)
            delta = profiling.metrics_delta(before, after)
            impact["effect"] = profiling.render_delta(delta)
            working = result.df
        elif result.status == "error":
            impact["effect"] = result.message[:160]
        else:
            impact["effect"] = "no data change (inspection only)"
        impacts.append(impact)
    return impacts


def apply_recorded_actions(df: pd.DataFrame, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministically replay a past run's effective actions on a new file.

    No LLM involved: each recorded ok-action goes back through the tool
    registry. Returns the new frame plus a line-by-line replay log.
    """
    from cleanai.agents.tools import registry

    working = df.copy()
    log: List[str] = []
    for record in actions:
        if record.get("status") != "ok":
            continue
        name = record["tool"]
        spec = registry.REGISTRY.get(name)
        if spec is None or not spec.mutates:
            continue
        before = profiling.snapshot(working)
        result = registry.execute(name, working, record.get("args") or {})
        if result.status == "ok" and result.mutated:
            delta = profiling.metrics_delta(before, profiling.snapshot(result.df))
            working = result.df
            log.append(f"{name}: {profiling.render_delta(delta)}")
        else:
            log.append(f"{name}: skipped ({result.message[:90]})")
    return {
        "df": working,
        "log": log,
        "score_before": profiling.quality_score(df),
        "score_after": profiling.quality_score(working),
    }


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

    # ---- review phase: suggestions in, plan out ----
    def state_values(self) -> Dict[str, Any]:
        return self.app.get_state(self.config).values

    def awaiting_review(self) -> bool:
        return bool(self.app.get_state(self.config).next)

    def get_suggestions(self) -> List[Dict[str, Any]]:
        return list(self.state_values().get("suggestions", []))

    def set_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        from cleanai.core import suggest

        self.app.update_state(self.config, {"suggestions": suggest.reindex(suggestions)})

    def add_user_request(self, request: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Turn a free-text user ask into suggestions and append them."""
        import json as json_module

        from langchain_core.messages import HumanMessage, SystemMessage

        from cleanai.agents.llm import build_llm, invoke_with_retry
        from cleanai.agents.prompts import SYSTEM_PROMPT
        from cleanai.agents.tools import registry
        from cleanai.core import suggest

        df = self.state_values()["current_df"]
        existing = self.get_suggestions()
        prompt = (
            "The user wants to add to the cleaning work. Convert their request into "
            'suggestions as JSON: {"suggestions": [{"title", "why", "severity", '
            '"tool", "columns", "params"}]}. Use only these tools:\n'
            f"{registry.tool_signatures()}\n\n"
            f"Columns available: {[str(c) for c in df.columns]}\n"
            f"Already planned (do not duplicate): "
            f"{json_module.dumps([{'tool': s['tool'], 'columns': s['columns']} for s in existing], default=str)}\n\n"
            f"User request: {request}\n\nJSON only."
        )
        response = invoke_with_retry(
            build_llm(json_mode=True, api_key=api_key),
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
        )
        new_items = suggest.parse_suggestions(
            response.content,
            valid_tools=list(registry.REGISTRY.keys()),
            valid_columns=[str(c) for c in df.columns],
            source="user",
        )
        combined = existing + [s.model_dump() for s in new_items]
        self.set_suggestions(combined)
        return [s.model_dump() for s in new_items]

    def approve(self, goal: str = "") -> List[Dict[str, Any]]:
        """Compile accepted suggestions into the plan and lock it in."""
        from cleanai.core import suggest

        plan = suggest.build_plan(self.get_suggestions())
        updates = {"plan_steps": plan, "plan_locked": True, "current_step_index": 0, "mode": "plan"}
        if goal.strip():
            updates["user_context"] = goal.strip()
            self.user_context = goal.strip()
        self.app.update_state(self.config, updates)
        return plan

    # legacy accessors still used by the advanced editor
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
            "tokens_used": values.get("tokens_used", 0),
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
