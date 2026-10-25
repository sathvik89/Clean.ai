"""Structured cleaning plans.

The planner LLM emits JSON; this module parses it defensively, validates
every step against the tool registry, and tracks execution progress.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PlanStep(BaseModel):
    id: str
    tool: str
    columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    priority: int = 2
    status: str = "pending"      # pending | running | done | blocked | skipped
    attempts: int = 0
    result: str = ""

    @field_validator("priority", mode="before")
    @classmethod
    def _clamp_priority(cls, v):
        try:
            return min(3, max(1, int(v)))
        except (TypeError, ValueError):
            return 2


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of possibly-noisy LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_plan(raw_text: str, valid_tools: List[str], valid_columns: List[str]) -> List[PlanStep]:
    """Parse planner output; silently drops steps that reference unknown tools."""
    payload = _extract_json(raw_text)
    if not payload:
        return []
    raw_steps = payload.get("steps", payload if isinstance(payload, list) else [])
    if not isinstance(raw_steps, list):
        return []

    steps: List[PlanStep] = []
    column_set = set(valid_columns)
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool", "")).strip().lower().removeprefix("tool_")
        if tool not in valid_tools:
            continue
        columns = [c for c in raw.get("columns", []) or [] if c in column_set]
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        steps.append(PlanStep(
            id=str(raw.get("id") or f"s{index + 1}"),
            tool=tool,
            columns=columns,
            params=params,
            rationale=str(raw.get("rationale", ""))[:140],
            priority=raw.get("priority", 2),
        ))
    # stable order: critical first, then original order
    steps.sort(key=lambda s: s.priority)
    for index, step in enumerate(steps):
        step.id = f"s{index + 1}"
    return steps


def render_progress(steps: List[Dict[str, Any]], current_index: int) -> str:
    if not steps:
        return "no plan steps"
    lines = []
    for index, raw in enumerate(steps):
        step = PlanStep(**raw)
        marker = {"done": "[done]", "blocked": "[blocked]", "skipped": "[skipped]"}.get(step.status, "")
        if index == current_index and step.status in ("pending", "running"):
            marker = "<- CURRENT"
        lines.append(f"{step.id} {step.tool} on {step.columns or 'table'} {marker}".rstrip())
    return "\n".join(lines)


def render_step(raw: Dict[str, Any]) -> str:
    step = PlanStep(**raw)
    parts = [f"{step.id}: call {step.tool}"]
    if step.columns:
        parts.append(f"on columns {step.columns}")
    if step.params:
        parts.append(f"with params {json.dumps(step.params, default=str)}")
    if step.rationale:
        parts.append(f"- rationale: {step.rationale}")
    if step.attempts:
        parts.append(f"(attempt {step.attempts + 1}, previous attempts failed - try different parameters)")
    return " ".join(parts)


def steps_to_dicts(steps: List[PlanStep]) -> List[Dict[str, Any]]:
    return [s.model_dump() for s in steps]
