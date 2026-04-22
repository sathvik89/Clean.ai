import os
import pickle
from typing import Any, Annotated, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langchain_core.tools import tool
from dotenv import load_dotenv

from state import AgentState
from nodes import (
    analyzer_node, planner_node, human_review_node, 
    executor_node, tool_handler_node, validator_node
)

load_dotenv()

# --- serialization logic ---
class PickleSerde(SerializerProtocol):
    """custom serializer to handle pandas dataframes using pickle."""
    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj)
    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("pickle", pickle.dumps(obj))
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, bytes_ = data
        if type_ == "pickle":
            return pickle.loads(bytes_)
        return pickle.loads(bytes_)

# --- tool wrappers for llm binding ---
@tool
def tool_get_summary():
    """gets a summary of the current data state."""
    pass # implementation handled in node

@tool
def tool_handle_missing(strategy: str, columns: Optional[List[str]] = None, fill_value: Any = None):
    """handles missing values using various strategies."""
    pass

@tool
def tool_remove_duplicates(columns: Optional[List[str]] = None):
    """removes duplicate rows based on columns."""
    pass

@tool
def tool_type_conversion(column_type_map: Dict[str, str]):
    """converts columns to specific types (int, float, datetime, etc.)."""
    pass

@tool
def tool_detect_outliers(columns: List[str]):
    """provides a report on outliers for specific columns."""
    pass

@tool
def tool_clean_text(columns: List[str], actions: List[str]):
    """cleans text data (lowercase, strip, remove special characters)."""
    pass

@tool
def tool_execute_custom_pandas(code: str):
    """executes custom pandas code on 'df'."""
    pass

@tool
def tool_handle_categorical(columns: List[str], method: str = 'label'):
    """encodes categorical variables using label or onehot encoding."""
    pass

all_tools = [
    tool_get_summary, tool_handle_missing, tool_remove_duplicates,
    tool_type_conversion, tool_detect_outliers, tool_clean_text,
    tool_execute_custom_pandas, tool_handle_categorical
]

# --- graph definition ---
workflow = StateGraph(AgentState)

# add nodes
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("executor", lambda x: executor_node(x, all_tools))
workflow.add_node("tools", tool_handler_node)
workflow.add_node("validator", validator_node)

# define edges
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "planner")
workflow.add_edge("planner", "human_review")
workflow.add_edge("human_review", "executor")

# conditional logic for execution loop
def router(state: AgentState):
    print(
        f"[clean.ai][router] executor routed with tool_calls="
        f"{len(state['messages'][-1].tool_calls) if state.get('messages') else 0}",
        flush=True,
    )
    if state["messages"][-1].tool_calls:
        return "tools"
    return "validator"

workflow.add_conditional_edges("executor", router)

def tool_loop_router(state: AgentState):
    tool_rounds = state.get("tool_rounds", 0)
    max_tool_rounds = state.get("max_tool_rounds", 6)
    print(
        f"[clean.ai][router] tool loop check tool_rounds={tool_rounds}/{max_tool_rounds}",
        flush=True,
    )
    if tool_rounds >= max_tool_rounds:
        return "validator"
    return "executor"

workflow.add_conditional_edges("tools", tool_loop_router)

# conditional logic for validation loop
def final_check(state: AgentState):
    validation_rounds = state.get("validation_rounds", 0)
    max_validation_rounds = state.get("max_validation_rounds", 6)
    print(
        f"[clean.ai][router] final check is_clean={state['is_clean']} "
        f"validation_rounds={validation_rounds}/{max_validation_rounds}",
        flush=True,
    )
    if state["is_clean"]:
        return END
    if state.get("run_error"):
        return END
    if validation_rounds >= max_validation_rounds:
        return END
    return "executor"

workflow.add_conditional_edges("validator", final_check)

# compile
memory = MemorySaver(serde=PickleSerde())
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])
