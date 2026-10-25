"""Graph state shared across all agent nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict, Union

import pandas as pd

from cleanai.core.profiling import ColumnCard


def append_logs(old: List[str], new: Union[str, List[str]]) -> List[str]:
    if isinstance(new, str):
        return old + [new]
    return old + new


def keep_recent_messages(old: List, new: List) -> List:
    """Messages only matter within one executor<->tools exchange, so the
    state keeps a short window instead of the whole conversation."""
    return (old + new)[-8:]


class AgentState(TypedDict):
    # inputs
    file_name: str
    user_context: str

    # data
    original_df: pd.DataFrame
    current_df: pd.DataFrame
    cards: List[ColumnCard]

    # agent outputs
    analysis_report: str
    plan_steps: List[Dict[str, Any]]
    current_step_index: int
    mode: str                      # "plan" | "repair"
    case_hints: str

    # episodic memory
    ledger: Annotated[List[Dict[str, Any]], operator.add]
    seen_calls: Annotated[List[str], operator.add]

    # validation
    is_clean: bool
    validation_feedback: str
    validation_signature: str
    stagnation_rounds: int
    run_error: str

    # loop counters
    tool_rounds: int
    validation_rounds: int

    logs: Annotated[List[str], append_logs]
    messages: Annotated[List, keep_recent_messages]


def initial_state(df: pd.DataFrame, file_name: str, user_context: str) -> AgentState:
    return AgentState(
        file_name=file_name,
        user_context=user_context or "general cleaning and formatting",
        original_df=df.copy(),
        current_df=df.copy(),
        cards=[],
        analysis_report="",
        plan_steps=[],
        current_step_index=0,
        mode="plan",
        case_hints="",
        ledger=[],
        seen_calls=[],
        is_clean=False,
        validation_feedback="",
        validation_signature="",
        stagnation_rounds=0,
        run_error="",
        tool_rounds=0,
        validation_rounds=0,
        logs=[],
        messages=[],
    )
