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
    tool_handler_node, validator_node,
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


def route_after_tools(state: AgentState) -> str:
    settings = get_settings()
    if state["tool_rounds"] >= settings.max_tool_rounds:
        return "validator"
    if state["mode"] == "repair":
        # re-check after every repair round so the run ends as soon as it's clean
        return "validator"
    if state["current_step_index"] >= len(state["plan_steps"]):
        return "validator"
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
    workflow.add_node("tools", tool_handler_node)
    workflow.add_node("validator", validator_node)

    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "eda")
    workflow.add_edge("eda", "human_review")
    workflow.add_edge("human_review", "executor")
    workflow.add_conditional_edges("executor", route_after_executor)
    workflow.add_conditional_edges("tools", route_after_tools)
    workflow.add_conditional_edges("validator", route_after_validator)

    if checkpointer is None:
        checkpointer = MemorySaver(serde=PickleSerde())
    interrupts = ["human_review"] if with_human_review else []
    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupts)
