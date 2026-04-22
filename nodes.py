from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import Dict
from tools import (
    get_data_summary, smart_handle_missing, rigorous_remove_duplicates,
    smart_type_conversion, detect_outliers_report, perform_text_cleaning,
    audit_data_quality, execute_custom_pandas, handle_categorical,
    get_validation_issues, infer_validation_profile, get_profile_validation_issues
)
from prompts import SYSTEM_PROMPT, ANALYZER_PROMPT, PLANNER_PROMPT, EXECUTOR_PROMPT
from state import AgentState
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def console_log(node_name: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[clean.ai][{timestamp}][{node_name}] {message}", flush=True)

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
    console_log("analyzer", f"starting analysis for {state['file_path']} with shape={state['current_df'].shape}")
    summary = get_data_summary(state["current_df"])
    validation_profile = infer_validation_profile(state["current_df"])
    prompt = ANALYZER_PROMPT.format(user_context=state["user_context"])
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=prompt), HumanMessage(content=summary)]
    response = get_llm().invoke(messages)
    
    # initialize work summary with the starting state
    initial_work = f"initial audit summary:\n{summary[:700]}"
    console_log("analyzer", "analysis complete")
    
    return {
        "analysis_report": response.content, 
        "validation_profile": validation_profile,
        "work_summary": initial_work,
        "logs": ["initial analysis complete"],
        "run_error": "",
    }

def planner_node(state: AgentState):
    """creates a detailed cleaning strategy."""
    console_log("planner", "building cleaning plan")
    prompt = PLANNER_PROMPT.format(
        user_context=state["user_context"],
        analysis_report=state["analysis_report"]
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=prompt)]
    response = get_llm().invoke(messages)
    console_log("planner", "plan ready")
    return {"cleaning_plan": response.content, "logs": ["strategic cleaning plan generated"]}

def human_review_node(state: AgentState):
    """marker node for human-in-the-loop."""
    return {"logs": ["awaiting human confirmation of the plan"]}

def executor_node(state: AgentState, tools: list):
    """agent executes cleaning tools with optimized history."""
    console_log(
        "executor",
        f"starting executor call with tool_rounds={state.get('tool_rounds', 0)} "
        f"validation_rounds={state.get('validation_rounds', 0)}",
    )
    prompt = EXECUTOR_PROMPT.format(
        user_context=state["user_context"],
        cleaning_plan=state["cleaning_plan"],
        validation_feedback=state.get("validation_feedback", "no unresolved validation issues yet"),
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
    tool_call_count = len(response.tool_calls) if getattr(response, "tool_calls", None) else 0
    console_log("executor", f"executor response received with tool_calls={tool_call_count}")
    return {"messages": [response]}

def tool_handler_node(state: AgentState):
    """manually handles tool calls and updates the dataframe state."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    current_df = state["current_df"].copy()
    new_logs = []
    tool_outputs = []
    updated_tool_rounds = state.get("tool_rounds", 0) + 1
    console_log("tools", f"processing tool round {updated_tool_rounds} with {len(last_message.tool_calls)} tool calls")
    
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        result = "success"
        console_log("tools", f"calling {name} with args={args}")
        
        try:
            # normalize name to handle potential llm variations
            norm_name = name.lower()
            if not norm_name.startswith("tool_"):
                norm_name = f"tool_{norm_name}"

            if norm_name == "tool_get_summary":
                result = get_data_summary(current_df)
            elif norm_name == "tool_handle_missing":
                before_nulls = {col: int(current_df[col].isnull().sum()) for col in args.get("columns", current_df.columns) if col in current_df.columns}
                current_df = smart_handle_missing(current_df, **args)
                after_nulls = {col: int(current_df[col].isnull().sum()) for col in before_nulls}
                details = []
                for col in before_nulls:
                    fixed_count = before_nulls[col] - after_nulls[col]
                    details.append(f"{col} fixed={fixed_count} remaining={after_nulls[col]}")
                new_logs.append(
                    f"handled missing values using {args.get('strategy', 'unknown')} on {list(before_nulls.keys())} | "
                    + "; ".join(details)
                )
            elif norm_name == "tool_remove_duplicates":
                before_rows = len(current_df)
                current_df = rigorous_remove_duplicates(current_df, **args)
                removed_count = before_rows - len(current_df)
                new_logs.append(
                    f"removed duplicates using columns={args.get('columns')} | removed_rows={removed_count}"
                )
            elif norm_name == "tool_type_conversion":
                before_types = {col: str(current_df[col].dtype) for col in args.get("column_type_map", {}) if col in current_df.columns}
                before_nulls = {col: int(current_df[col].isnull().sum()) for col in before_types}
                current_df = smart_type_conversion(current_df, **args)
                details = []
                for col, old_type in before_types.items():
                    new_type = str(current_df[col].dtype)
                    new_nulls = int(current_df[col].isnull().sum())
                    details.append(f"{col}: {old_type} -> {new_type}, nulls={before_nulls[col]} -> {new_nulls}")
                new_logs.append("converted types | " + "; ".join(details))
            elif norm_name == "tool_detect_outliers":
                result = detect_outliers_report(current_df, **args)
                new_logs.append(f"analyzed outliers for columns={args.get('columns')} | report={result}")
            elif norm_name in ["tool_clean_text", "tool_perform_text_cleaning"]:
                current_df = perform_text_cleaning(current_df, **args)
                new_logs.append(
                    f"cleaned text on columns={args.get('columns')} | actions={args.get('actions')}"
                )
            elif norm_name == "tool_execute_custom_pandas":
                current_df = execute_custom_pandas(current_df, **args)
                new_logs.append("executed custom pandas transformation")
            elif norm_name == "tool_handle_categorical":
                current_df = handle_categorical(current_df, **args)
                new_logs.append(
                    f"handled categorical encoding on columns={args.get('columns')} | method={args.get('method', 'label')}"
                )
            else:
                result = f"error: tool '{name}' is not recognized. please use only supported tools."
                new_logs.append(f"unknown tool call attempted: {name}")
                console_log("tools", f"unknown tool: {name}")
        except Exception as e:
            result = f"error: {str(e)}"
            new_logs.append(f"tool failed ({name}): {str(e)}")
            console_log("tools", f"{name} failed with error={str(e)}")
            
        tool_outputs.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
        
    console_log("tools", f"tool round {updated_tool_rounds} complete")
    return {
        "current_df": current_df,
        "df_history": [state["current_df"]], # store previous for versioning
        "messages": tool_outputs,
        "logs": new_logs,
        "tool_rounds": updated_tool_rounds,
    }

def validator_node(state: AgentState):
    """checks if the cleaning goals have been met."""
    next_validation_round = state.get("validation_rounds", 0) + 1
    console_log("validator", f"starting validation round {next_validation_round}")
    audit = audit_data_quality(state["current_df"])
    issues = get_validation_issues(state["current_df"])
    issues.extend(get_profile_validation_issues(state["current_df"], state.get("validation_profile", {})))
    issues = list(dict.fromkeys(issues))
    is_clean = len(issues) == 0
    validation_feedback = "all validation checks passed"
    if issues:
        validation_feedback = "remaining issues: " + " | ".join(issues)
    
    # update work summary with latest findings
    current_summary = state.get("work_summary", "")
    new_step_info = f"\n- iteration audit: {audit.get('remaining_nulls', 'no nulls')} left; {audit.get('duplicates', 0)} duplicates."
    
    # keep the work summary from growing too large itself
    updated_summary = (current_summary + new_step_info)[-1400:]
    previous_signature = state.get("validation_signature", "")
    current_signature = " | ".join(sorted(issues))
    stagnation_rounds = state.get("stagnation_rounds", 0)
    if not is_clean:
        if current_signature == previous_signature and current_signature:
            stagnation_rounds += 1
        else:
            stagnation_rounds = 0
    else:
        stagnation_rounds = 0
    run_error = ""
    if stagnation_rounds >= state.get("max_stagnation_rounds", 2) and not is_clean:
        run_error = "stopped because the same validation issues repeated without progress"
        console_log("validator", run_error)
    elif next_validation_round >= state.get("max_validation_rounds", 6) and not is_clean:
        run_error = "stopped after reaching the maximum validation rounds"
        console_log("validator", run_error)
    console_log("validator", f"validation round {next_validation_round} complete, is_clean={is_clean}")
    
    return {
        "is_clean": is_clean,
        "audit_report": {**audit, "validation_issues": issues},
        "work_summary": updated_summary,
        "validation_feedback": validation_feedback,
        "logs": [f"data quality audit complete: {validation_feedback}"],
        "validation_rounds": next_validation_round,
        "validation_signature": current_signature,
        "stagnation_rounds": stagnation_rounds,
        "run_error": run_error,
    }
