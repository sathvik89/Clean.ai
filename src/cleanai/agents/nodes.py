"""Graph nodes: analyzer, planner, executor, tool handler, validator.

Design principle (from the memory architecture): the executor never sees
raw conversation history. Every turn it gets a freshly rendered view of
structured state - plan progress, column cards, the action ledger and
the last tool result - which keeps the context small and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from cleanai.agents import prompts
from cleanai.agents.llm import build_llm, invoke_with_retry
from cleanai.agents.state import AgentState
from cleanai.agents.tools import registry
from cleanai.config import get_settings
from cleanai.core import plan as planning
from cleanai.core import profiling
from cleanai.core.memory import (
    ActionRecord, CaseMemory, dataset_fingerprint, render_cards, render_ledger,
)


def log_line(node: str, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[clean.ai][{stamp}][{node}] {message}", flush=True)


def get_case_memory() -> CaseMemory:
    return CaseMemory(get_settings().cases_db_path)


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------

def analyzer_node(state: AgentState) -> Dict[str, Any]:
    df = state["current_df"]
    log_line("analyzer", f"profiling {state['file_name']} shape={df.shape}")
    cards = profiling.build_cards(df)
    profile_text = profiling.summarize_for_llm(df, cards)

    hints = ""
    try:
        hints = get_case_memory().hints_for(dataset_fingerprint(cards))
    except Exception as exc:
        log_line("analyzer", f"case memory unavailable: {exc}")

    prompt = prompts.ANALYZER_PROMPT.format(
        user_context=state["user_context"], profile=profile_text,
    )
    response = invoke_with_retry(
        build_llm(),
        [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=prompt)],
    )
    log_line("analyzer", "analysis report ready")
    return {
        "cards": cards,
        "analysis_report": response.content,
        "case_hints": hints,
        "logs": ["analysis complete: " + f"{len(profiling.all_open_issues(df, cards))} issues found"],
        "run_error": "",
    }


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def fallback_plan(df, cards) -> List[planning.PlanStep]:
    """Deterministic plan used when the LLM's JSON cannot be parsed."""
    steps: List[planning.PlanStep] = []
    type_map = {}
    for card in cards:
        kinds = {i.kind for i in card.issues_open}
        if "wrong_dtype" in kinds or "non_numeric_values" in kinds:
            type_map[card.name] = "float" if card.role == "numeric" else card.role
        elif "invalid_dates" in kinds:
            type_map[card.name] = "datetime"
    if type_map:
        steps.append(planning.PlanStep(
            id="s1", tool="convert_types", params={"column_type_map": type_map},
            rationale="fix wrong dtypes first", priority=1,
        ))
    if int(df.duplicated().sum()) > 0:
        steps.append(planning.PlanStep(
            id="s2", tool="remove_duplicates", rationale="drop exact duplicate rows", priority=1,
        ))
    null_cols = [c.name for c in cards if c.null_count > 0]
    if null_cols:
        steps.append(planning.PlanStep(
            id="s3", tool="handle_missing", columns=null_cols,
            params={"strategy": "auto"}, rationale="fill remaining nulls", priority=2,
        ))
    text_cols = [c.name for c in cards if c.role in ("text", "categorical")]
    if text_cols:
        steps.append(planning.PlanStep(
            id="s4", tool="clean_text", columns=text_cols,
            params={"actions": ["strip", "collapse_spaces"]},
            rationale="standardize text", priority=3,
        ))
    for index, step in enumerate(steps):
        step.id = f"s{index + 1}"
    return steps


def planner_node(state: AgentState) -> Dict[str, Any]:
    log_line("planner", "building structured plan")
    df = state["current_df"]
    cards = state["cards"]
    hints_block = f"\nPAST-RUN HINTS:\n{state['case_hints']}" if state.get("case_hints") else ""
    prompt = prompts.PLANNER_PROMPT.format(
        tool_signatures=registry.tool_signatures(),
        user_context=state["user_context"],
        analysis_report=state["analysis_report"],
        profile=profiling.summarize_for_llm(df, cards),
        case_hints=hints_block,
    )
    response = invoke_with_retry(
        build_llm(json_mode=True),
        [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=prompt)],
    )
    steps = planning.parse_plan(
        response.content,
        valid_tools=list(registry.REGISTRY.keys()),
        valid_columns=[str(c) for c in df.columns],
    )
    source = "llm"
    if not steps:
        steps = fallback_plan(df, cards)
        source = "fallback"
        log_line("planner", "llm plan unusable, using deterministic fallback")
    log_line("planner", f"plan ready with {len(steps)} steps ({source})")
    return {
        "plan_steps": planning.steps_to_dicts(steps),
        "current_step_index": 0,
        "mode": "plan",
        "logs": [f"cleaning plan ready: {len(steps)} steps"],
    }


def human_review_node(state: AgentState) -> Dict[str, Any]:
    return {"logs": ["awaiting human approval of the plan"]}


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

def _last_tool_result_block(state: AgentState) -> str:
    settings = get_settings()
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            content = str(message.content)
            if len(content) > settings.tool_output_max_chars:
                content = content[: settings.tool_output_max_chars] + "\n...(clipped)"
            return f"\nLAST TOOL RESULT:\n{content}\n"
    return "\n"


def executor_node(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    log_line(
        "executor",
        f"mode={state['mode']} step={state['current_step_index']}/{len(state['plan_steps'])} "
        f"tool_rounds={state['tool_rounds']}",
    )
    cards_text = render_cards(
        state["cards"], max_cards=settings.cards_max_per_prompt,
        only_dirty=(state["mode"] == "repair"),
    )
    ledger_text = render_ledger(state["ledger"], settings.ledger_max_entries)
    last_result = _last_tool_result_block(state)

    if state["mode"] == "repair":
        prompt = prompts.EXECUTOR_REPAIR_PROMPT.format(
            validation_feedback=state.get("validation_feedback", "none"),
            cards=cards_text, ledger=ledger_text, last_result=last_result,
        )
    else:
        steps = state["plan_steps"]
        index = state["current_step_index"]
        current = planning.render_step(steps[index]) if index < len(steps) else "all steps finished"
        prompt = prompts.EXECUTOR_STEP_PROMPT.format(
            plan_progress=planning.render_progress(steps, index),
            current_step=current,
            cards=cards_text, ledger=ledger_text, last_result=last_result,
        )

    llm = build_llm().bind_tools(registry.langchain_tools())
    response = invoke_with_retry(
        llm,
        [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=prompt)],
    )
    call_count = len(response.tool_calls) if getattr(response, "tool_calls", None) else 0
    log_line("executor", f"model proposed {call_count} tool call(s)")
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# tool handler
# ---------------------------------------------------------------------------

def _call_fingerprint(tool: str, args: Dict[str, Any], df_sig: str) -> str:
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    return hashlib.sha1(f"{canonical}|{df_sig}".encode()).hexdigest()[:12]


def tool_handler_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return {}

    settings = get_settings()
    df = state["current_df"]
    steps = [dict(s) for s in state["plan_steps"]]
    step_index = state["current_step_index"]
    seen = set(state["seen_calls"])
    ledger_seq = len(state["ledger"])

    new_records: List[Dict[str, Any]] = []
    new_seen: List[str] = []
    new_logs: List[str] = []
    tool_messages: List[ToolMessage] = []

    for call in last_message.tool_calls:
        name = registry.normalize_tool_name(call["name"])
        args = call.get("args") or {}
        df_sig = profiling.df_signature(df)
        fingerprint = _call_fingerprint(name, args, df_sig)

        if fingerprint in seen:
            message = (
                f"DUPLICATE: {name} was already run with these exact arguments on "
                "identical data and changed nothing. Choose a different action."
            )
            tool_messages.append(ToolMessage(tool_call_id=call["id"], content=message))
            new_logs.append(f"skipped duplicate call to {name}")
            log_line("tools", f"blocked duplicate {name}")
            continue

        log_line("tools", f"running {name} args={json.dumps(args, default=str)[:160]}")
        before = profiling.snapshot(df)
        result = registry.execute(name, df, args)
        ledger_seq += 1

        delta_text = ""
        if result.mutated:
            after = profiling.snapshot(result.df)
            delta = profiling.metrics_delta(before, after)
            delta_text = profiling.render_delta(delta)
            if delta:
                df = result.df
            else:
                result.status = "noop" if result.status == "ok" else result.status
        record = ActionRecord(
            seq=ledger_seq, tool=name, args=args,
            status=result.status if result.status != "blocked" else "ok",
            delta=delta_text,
            error=result.message if result.status == "error" else "",
            df_signature=profiling.df_signature(df),
        )
        new_records.append(record.to_dict())
        seen.add(fingerprint)
        new_seen.append(fingerprint)
        new_logs.append(record.render())

        content = result.message
        if result.mutated:
            content = f"{result.message} | effect: {delta_text or 'NO CHANGE - data is identical'}"
        if len(content) > settings.tool_output_max_chars:
            content = content[: settings.tool_output_max_chars] + "\n...(clipped)"
        tool_messages.append(ToolMessage(tool_call_id=call["id"], content=content))

        # plan step bookkeeping
        if state["mode"] == "plan" and step_index < len(steps):
            step = steps[step_index]
            spec = registry.REGISTRY.get(name)
            if result.status == "blocked":
                step["status"] = "blocked"
                step["result"] = result.message[:140]
                step_index += 1
            elif result.status == "error" or result.status == "noop":
                step["attempts"] = step.get("attempts", 0) + 1
                if step["attempts"] >= settings.max_step_attempts:
                    step["status"] = "blocked"
                    step["result"] = "gave up after repeated failures"
                    step_index += 1
            elif spec and spec.mutates and result.status == "ok":
                step["status"] = "done"
                step["result"] = delta_text[:140]
                step_index += 1
            # read-only tools leave the step as-is

    return {
        "current_df": df,
        "cards": profiling.refresh_cards(df, state["cards"]),
        "plan_steps": steps,
        "current_step_index": step_index,
        "ledger": new_records,
        "seen_calls": new_seen,
        "messages": tool_messages,
        "logs": new_logs,
        "tool_rounds": state["tool_rounds"] + 1,
    }


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def validator_node(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    round_number = state["validation_rounds"] + 1
    log_line("validator", f"validation round {round_number}")

    df = state["current_df"]
    cards = profiling.refresh_cards(df, state["cards"])
    issues = profiling.all_open_issues(df, cards)
    is_clean = not issues

    feedback = "all validation checks passed"
    if issues:
        feedback = "remaining issues: " + " | ".join(i.describe() for i in issues[:12])

    signature = " | ".join(sorted(i.describe() for i in issues))
    stagnation = state["stagnation_rounds"]
    if not is_clean and signature == state["validation_signature"] and signature:
        stagnation += 1
    elif is_clean:
        stagnation = 0
    else:
        stagnation = 0

    run_error = ""
    if not is_clean:
        if stagnation >= settings.max_stagnation_rounds:
            run_error = "stopped: the same validation issues keep repeating without progress"
        elif round_number >= settings.max_validation_rounds:
            run_error = "stopped: reached the maximum number of validation rounds"
        elif state["tool_rounds"] >= settings.max_tool_rounds:
            run_error = "stopped: reached the maximum number of tool rounds"
    if run_error:
        log_line("validator", run_error)
    log_line("validator", f"is_clean={is_clean} ({len(issues)} issues)")

    return {
        "cards": cards,
        "is_clean": is_clean,
        "validation_feedback": feedback,
        "validation_signature": signature,
        "stagnation_rounds": stagnation,
        "validation_rounds": round_number,
        "run_error": run_error,
        "mode": "repair",
        "logs": [f"validation round {round_number}: {feedback}"],
    }
