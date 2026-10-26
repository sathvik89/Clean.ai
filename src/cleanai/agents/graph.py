"""Graph wiring: analyzer -> planner -> human review -> executor <-> tools -> validator.

The validator can bounce the run back to the executor in repair mode
until the data is clean or a guardrail (round caps, stagnation) stops it.
"""

from __future__ import annotations

import pickle
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import END, StateGraph

from cleanai.agents.nodes import (
    analyzer_node, eda_node, executor_node, human_review_node,
    mech_node, step_fully_specified, tool_handler_node, validator_node,
)
from cleanai.agents.state import AgentState
from cleanai.config import get_settings


class PickleSerde(SerializerProtocol):
    """Checkpoint serializer that can handle pandas DataFrames."""

    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("pickle", pickle.dumps(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return pickle.loads(data[1])


def route_after_executor(state: AgentState) -> str:
    if state["messages"] and getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "validator"


def route_next_step(state: AgentState) -> str:
    """After review, mech or tools: decide who handles the next plan step.

    Fully-specified steps run mechanically (no LLM, no drift); only steps
    that still need judgment or parameters go to the model.
    """
    settings = get_settings()
    if state["tool_rounds"] >= settings.max_tool_rounds:
        return "validator"
    if state["mode"] == "repair":
        # re-check after every repair round so the run ends as soon as it's clean
        return "validator"
    index = state["current_step_index"]
    steps = state["plan_steps"]
    if index >= len(steps):
        return "validator"
    if step_fully_specified(steps[index]):
        return "mech"
    return "executor"


def route_after_validator(state: AgentState) -> str:
    if state["is_clean"] or state["run_error"]:
        return END
    return "executor"


def build_graph(checkpointer=None, with_human_review: bool = True):
    workflow = StateGraph(AgentState)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("mech", mech_node)
    workflow.add_node("tools", tool_handler_node)
    workflow.add_node("validator", validator_node)

    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "eda")
    workflow.add_edge("eda", "human_review")
    workflow.add_conditional_edges("human_review", route_next_step)
    workflow.add_conditional_edges("mech", route_next_step)
    workflow.add_conditional_edges("executor", route_after_executor)
    workflow.add_conditional_edges("tools", route_next_step)
    workflow.add_conditional_edges("validator", route_after_validator)

    if checkpointer is None:
        checkpointer = MemorySaver(serde=PickleSerde())
    interrupts = ["human_review"] if with_human_review else []
    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupts)
