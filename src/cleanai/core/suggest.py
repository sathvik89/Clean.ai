"""Cleaning suggestions: the contract between the agent's EDA and the user.

The agent proposes suggestions; the user toggles them and adds their own.
Accepted suggestions compile deterministically into the execution plan -
no second LLM call, no drift between what was approved and what runs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from cleanai.core.plan import PlanStep, _extract_json, steps_to_dicts


class Suggestion(BaseModel):
    id: str
    title: str
    why: str = ""
    severity: str = "medium"          # high | medium | low
    tool: str
    columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    source: str = "agent"             # agent | user
    accepted: bool = True


# execution order: structure first, then values, then cosmetics
TOOL_ORDER = {
    "normalize_nulls": 0,
    "rename_columns": 0,
    "extract_numbers": 1,
    "parse_dates": 1,
    "convert_types": 1,
    "replace_values": 2,
    "remove_duplicates": 3,
    "standardize_categories": 4,
    "fix_outliers": 5,
    "flag_rows": 5,
    "run_pandas": 5,
    "impute_group": 6,
    "handle_missing": 6,
    "clean_text": 7,
    "drop_columns": 8,
    "encode_categorical": 9,
}

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def parse_suggestions(
    raw_text: str, valid_tools: List[str], valid_columns: List[str], source: str = "agent",
) -> List[Suggestion]:
    payload = _extract_json(raw_text)
    if not payload:
        return []
    raw_items = payload.get("suggestions", payload if isinstance(payload, list) else [])
    if not isinstance(raw_items, list):
        return []

    column_set = set(valid_columns)
    suggestions: List[Suggestion] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool", "")).strip().lower().removeprefix("tool_")
        if tool not in valid_tools:
            continue
        severity = str(raw.get("severity", "medium")).lower()
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        suggestions.append(Suggestion(
            id=f"{source[0]}{index + 1}",
            title=str(raw.get("title", tool.replace("_", " ")))[:90],
            why=str(raw.get("why", ""))[:220],
            severity=severity if severity in SEVERITY_RANK else "medium",
            tool=tool,
            columns=[c for c in raw.get("columns", []) or [] if c in column_set],
            params=params,
            source=source,
        ))
    return suggestions


def reindex(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index, item in enumerate(suggestions):
        item["id"] = f"{item.get('source', 'agent')[0]}{index + 1}"
    return suggestions


# these tools legitimately create NEW missing values (coercions, nullify),
# so a run containing them always ends with a catch-all fill
NULL_REVEALING_TOOLS = {"normalize_nulls", "convert_types", "parse_dates",
                        "extract_numbers", "fix_outliers"}


def build_plan(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compile accepted suggestions into ordered plan steps."""
    accepted = [Suggestion(**s) for s in suggestions if s.get("accepted", True)]
    accepted.sort(key=lambda s: (TOOL_ORDER.get(s.tool, 5), SEVERITY_RANK.get(s.severity, 1)))
    steps = []
    for index, item in enumerate(accepted):
        steps.append(PlanStep(
            id=f"s{index + 1}",
            tool=item.tool,
            columns=item.columns,
            params=item.params,
            rationale=item.title[:140],
            priority=SEVERITY_RANK.get(item.severity, 1) + 1,
        ))

    reveals_nulls = any(s.tool in NULL_REVEALING_TOOLS for s in accepted)
    has_catch_all = any(s.tool == "handle_missing" and not s.columns for s in accepted)
    if accepted and reveals_nulls and not has_catch_all:
        steps.append(PlanStep(
            id=f"s{len(steps) + 1}",
            tool="handle_missing",
            params={"strategy": "auto"},
            rationale="safety net: fill values the earlier fixes revealed as missing",
            priority=3,
        ))
    return steps_to_dicts(steps)


def fallback_suggestions(df, cards) -> List[Suggestion]:
    """Deterministic suggestions straight from detected issues, used when
    the model's JSON is unusable."""
    from cleanai.core.profiling import detect_table_issues

    suggestions: List[Suggestion] = []
    seen_kinds = set()

    def add(tool, title, why, severity, columns=None, params=None):
        suggestions.append(Suggestion(
            id=f"a{len(suggestions) + 1}", title=title, why=why, severity=severity,
            tool=tool, columns=columns or [], params=params or {},
        ))

    disguised = [c.name for c in cards if any(i.kind == "disguised_nulls" for i in c.issues_open)]
    if disguised:
        add("normalize_nulls", "Convert disguised null text to real missing values",
            "values like 'N/A' and '-' hide from missing-value handling", "high", disguised)

    type_map = {}
    for card in cards:
        kinds = {i.kind for i in card.issues_open}
        if "non_numeric_values" in kinds or ("wrong_dtype" in kinds and card.role == "numeric"):
            type_map[card.name] = "float"
    if type_map:
        add("convert_types", "Convert numeric text columns to numbers",
            "numbers stored as text break every calculation", "high",
            params={"column_type_map": type_map})

    date_cols = [c.name for c in cards if any(i.kind in ("invalid_dates", "wrong_dtype") for i in c.issues_open) and c.role == "datetime"]
    if date_cols:
        add("parse_dates", "Parse messy date columns", "mixed date formats detected", "high", date_cols)

    if any(i.kind == "duplicate_rows" for i in detect_table_issues(df)):
        add("remove_duplicates", "Remove exact duplicate rows", "identical rows inflate counts", "high")

    null_cols = [c.name for c in cards if c.null_count > 0]
    if null_cols:
        add("handle_missing", "Fill remaining missing values",
            "median for numeric, most-frequent for categorical", "medium",
            null_cols, {"strategy": "auto"})

    text_cols = [c.name for c in cards if c.role in ("text", "categorical")]
    if text_cols:
        add("clean_text", "Standardize text whitespace",
            "stray spaces create fake category variants", "low",
            text_cols, {"actions": ["strip", "collapse_spaces"]})
    return suggestions


def narrative_to_markdown(narrative: Any) -> str:
    """Render the agent's EDA narrative JSON into readable markdown."""
    if isinstance(narrative, str):
        return narrative
    if not isinstance(narrative, dict):
        return ""
    parts = []
    if narrative.get("overview"):
        parts.append(str(narrative["overview"]))
    sections = [("key_findings", "Key findings"), ("risks", "Data quality risks"),
                ("relationships", "Relationships worth knowing")]
    for key, heading in sections:
        items = narrative.get(key) or []
        if isinstance(items, str):
            items = [items]
        if items:
            bullets = "\n".join(f"- {str(item)}" for item in items[:8])
            parts.append(f"**{heading}**\n\n{bullets}")
    return "\n\n".join(parts)
