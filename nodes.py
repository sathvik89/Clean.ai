from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import Annotated, Dict, Any, List, Optional
import pandas as pd
from tools import (
    get_data_summary, smart_handle_missing, rigorous_remove_duplicates,
    smart_type_conversion, detect_outliers_report, perform_text_cleaning,
    audit_data_quality, execute_custom_pandas, handle_categorical
)
from prompts import SYSTEM_PROMPT, ANALYZER_PROMPT, PLANNER_PROMPT, EXECUTOR_PROMPT, VALIDATOR_PROMPT
from state import AgentState
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """factory function to initialize the llm with current environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # return a dummy or wait for the environment to be set
        # since we are in a streamlit app, the key will be set in the ui
        api_key = "placeholder_key_will_be_overridden_in_ui"
        
    return ChatGroq(
        temperature=0, 
        model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        groq_api_key=api_key
    )

def analyzer_node(state: AgentState):
    """analyzes data and sets up the initial work summary."""
    summary = get_data_summary(state["current_df"])
    prompt = ANALYZER_PROMPT.format(user_context=state["user_context"])
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=prompt), HumanMessage(content=summary)]
    response = get_llm().invoke(messages)
    
    # initialize work summary with the starting state
    initial_work = f"### initial audit:\n{summary[:500]}... (truncated for brevity)"
    
    return {
        "analysis_report": response.content, 
        "work_summary": initial_work,
        "logs": ["initial analysis complete"]
    }

def planner_node(state: AgentState):
    """creates a detailed cleaning strategy."""
    prompt = PLANNER_PROMPT.format(
        user_context=state["user_context"],
        analysis_report=state["analysis_report"]
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=prompt)]
    response = get_llm().invoke(messages)
    return {"cleaning_plan": response.content, "logs": ["strategic cleaning plan generated"]}

def human_review_node(state: AgentState):
    """marker node for human-in-the-loop."""
    return {"logs": ["awaiting human confirmation of the plan"]}

def executor_node(state: AgentState, tools: list):
    """agent executes cleaning tools with optimized history."""
    prompt = EXECUTOR_PROMPT.format(
        user_context=state["user_context"],
        cleaning_plan=state["cleaning_plan"]
    )
    
    # create a summary message to keep context without bloating history
    context_summary = f"### summary of work done so far:\n{state.get('work_summary', 'no actions taken yet.')}"
    
    # only keep the last 4 messages (2 turns) to stay within token limits
    recent_messages = state["messages"][-4:] if state.get("messages") else []
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT), 
        SystemMessage(content=prompt),
        SystemMessage(content=context_summary)
    ] + recent_messages
    
    if not any(isinstance(m, HumanMessage) for m in recent_messages):
        messages.append(HumanMessage(content="continue with the cleaning plan based on the latest state."))
        
    llm_with_tools = get_llm().bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_handler_node(state: AgentState):
    """manually handles tool calls and updates the dataframe state."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    current_df = state["current_df"].copy()
    new_logs = []
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        result = "success"
        
        try:
            if name == "tool_get_summary":
                result = get_data_summary(current_df)
            elif name == "tool_smart_handle_missing":
                current_df = smart_handle_missing(current_df, **args)
                new_logs.append(f"handled missing values: {args}")
            elif name == "tool_remove_duplicates":
                current_df = rigorous_remove_duplicates(current_df, **args)
                new_logs.append(f"removed duplicates: {args}")
            elif name == "tool_smart_type_conversion":
                current_df = smart_type_conversion(current_df, **args)
                new_logs.append(f"converted types: {args}")
            elif name == "tool_detect_outliers":
                result = detect_outliers_report(current_df, **args)
                new_logs.append(f"analyzed outliers: {args}")
            elif name == "tool_perform_text_cleaning":
                current_df = perform_text_cleaning(current_df, **args)
                new_logs.append(f"cleaned text: {args}")
            elif name == "tool_execute_custom_pandas":
                current_df = execute_custom_pandas(current_df, **args)
                new_logs.append(f"executed custom pandas: {args['code']}")
            elif name == "tool_handle_categorical":
                current_df = handle_categorical(current_df, **args)
                new_logs.append(f"handled categorical encoding: {args}")
        except Exception as e:
            result = f"error: {str(e)}"
            new_logs.append(f"tool failed ({name}): {str(e)}")
            
        tool_outputs.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
        
    return {
        "current_df": current_df,
        "df_history": [state["current_df"]], # store previous for versioning
        "messages": tool_outputs,
        "logs": new_logs
    }

def validator_node(state: AgentState):
    """checks if the cleaning goals have been met."""
    audit = audit_data_quality(state["current_df"])
    prompt = VALIDATOR_PROMPT.format(
        user_context=state["user_context"],
        audit_report=str(audit)
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=prompt)]
    response = get_llm().invoke(messages)
    
    is_clean = "is_clean: true" in response.content.lower()
    
    # update work summary with latest findings
    current_summary = state.get("work_summary", "")
    new_step_info = f"\n- iteration audit: {audit.get('remaining_nulls', 'no nulls')} left; {audit.get('duplicates', 0)} duplicates."
    
    # keep the work summary from growing too large itself
    updated_summary = (current_summary + new_step_info)[-2000:]
    
    return {
        "is_clean": is_clean,
        "audit_report": audit,
        "work_summary": updated_summary,
        "logs": ["data quality audit complete"]
    }
