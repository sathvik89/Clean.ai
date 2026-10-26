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
from cleanai.agents.llm import build_llm, invoke_with_retry, usage_tokens
from cleanai.agents.state import AgentState
from cleanai.agents.tools import registry
from cleanai.config import get_settings
from cleanai.core import eda, plan as planning, profiling, suggest
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
    bundle = eda.compute_bundle(df, cards)

    hints = ""
    try:
        hints = get_case_memory().hints_for(dataset_fingerprint(cards))
    except Exception as exc:
        log_line("analyzer", f"case memory unavailable: {exc}")

    issues = profiling.all_open_issues(df, cards)
    log_line("analyzer", f"profile ready, {len(issues)} issues detected")
    return {
        "cards": cards,
        "eda_bundle": bundle,
        "case_hints": hints,
        "logs": [f"profiled the data: {len(issues)} issues detected"],
        "run_error": "",
    }


# ---------------------------------------------------------------------------
# eda agent
# ---------------------------------------------------------------------------

def eda_node(state: AgentState) -> Dict[str, Any]:
    """Agent-driven exploration: investigate with read-only tools, then
    produce a narrative report plus cleaning suggestions."""
    settings = get_settings()
    df = state["current_df"]
    cards = state["cards"]
    profile_text = profiling.summarize_for_llm(df, cards)
    eda_summary = eda.render_for_llm(state["eda_bundle"])
    tokens = 0

    # phase 1: free investigation with inspection tools (max 3 rounds)
    log_line("eda", "agent is exploring the data")
    inspect_tools = registry.langchain_tools(registry.INSPECTION_TOOLS)
    llm = build_llm().bind_tools(inspect_tools)
    messages = [
        SystemMessage(content=prompts.SYSTEM_PROMPT),
        HumanMessage(content=prompts.EDA_INVESTIGATE_PROMPT.format(
            profile=profile_text, eda_summary=eda_summary)),
    ]
    investigation_notes = []
    final_note = ""
    for round_number in range(3):
        response = invoke_with_retry(llm, messages)
        tokens += usage_tokens(response)
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            final_note = str(response.content)
            break
        messages.append(response)
        log_line("eda", f"investigation round {round_number + 1}: {len(calls)} lookups")
        for call in calls:
            name = registry.normalize_tool_name(call["name"])
            if name not in registry.INSPECTION_TOOLS:
                result_text = f"error: {name} is not available during exploration"
            else:
                result_text = registry.execute(name, df, call.get("args") or {}).message
            clipped = result_text[: settings.tool_output_max_chars // 2]
            investigation_notes.append(f"{name}({json.dumps(call.get('args') or {}, default=str)[:80]}):\n{clipped}")
            messages.append(ToolMessage(tool_call_id=call["id"], content=clipped))

    investigation = final_note or "(no extra notes)"
    if investigation_notes:
        investigation += "\n\nlookups made:\n" + "\n".join(n[:400] for n in investigation_notes[-6:])

    # phase 2: structured report + suggestions
    log_line("eda", "writing the eda report and suggestions")
    hints_block = f"\nHINTS FROM SIMILAR PAST RUNS:\n{state['case_hints']}" if state.get("case_hints") else ""
    report_prompt = prompts.EDA_REPORT_PROMPT.format(
        tool_signatures=registry.tool_signatures(),
        user_context=state.get("user_context", ""),
        profile=profile_text,
        eda_summary=eda_summary,
        investigation=investigation[:3000],
        case_hints=hints_block,
    )
    response = invoke_with_retry(
        build_llm(json_mode=True),
        [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=report_prompt)],
    )
    tokens += usage_tokens(response)

    payload = planning._extract_json(response.content) or {}
    narrative_md = suggest.narrative_to_markdown(payload.get("narrative"))
    suggestions = suggest.parse_suggestions(
        response.content,
        valid_tools=list(registry.REGISTRY.keys()),
        valid_columns=[str(c) for c in df.columns],
    )
    source = "llm"
    if not suggestions:
        suggestions = suggest.fallback_suggestions(df, cards)
        source = "fallback"
        log_line("eda", "model suggestions unusable, using deterministic fallback")
    if not narrative_md:
        narrative_md = "The automated profile below describes the dataset; the model's narrative was unavailable."
    log_line("eda", f"report ready with {len(suggestions)} suggestions ({source})")

    return {
        "tokens_used": tokens,
        "analysis_report": narrative_md,
        "suggestions": [s.model_dump() for s in suggestions],
        "logs": [f"eda complete: report written, {len(suggestions)} cleaning suggestions proposed"],
    }


def human_review_node(state: AgentState) -> Dict[str, Any]:
    if state.get("plan_locked") or state.get("plan_steps"):
        return {"logs": ["plan approved, starting execution"]}
    # headless runs (cli --yes, tests) compile the plan from all suggestions
    plan = suggest.build_plan(state.get("suggestions", []))
    return {
        "plan_steps": plan,
        "current_step_index": 0,
        "logs": [f"plan compiled from {len(plan)} accepted suggestions"],
    }


# ---------------------------------------------------------------------------
# mechanical executor: approved steps with complete params never need the LLM
# ---------------------------------------------------------------------------

def step_args(step: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(step.get("params") or {})
    if step.get("columns"):
        args.setdefault("columns", step["columns"])
    return args


def step_fully_specified(step: Dict[str, Any]) -> bool:
    spec = registry.REGISTRY.get(step.get("tool", ""))
    if spec is None:
        return False
    try:
        spec.args_model(**step_args(step))
        return True
    except Exception:
        return False


def mech_node(state: AgentState) -> Dict[str, Any]:
    """Execute the current plan step exactly as approved - no model, no drift."""
    settings = get_settings()
    df = state["current_df"]
    steps = [dict(s) for s in state["plan_steps"]]
    index = state["current_step_index"]
    step = steps[index]
    args = step_args(step)
    log_line("mech", f"running approved step {step['id']} {step['tool']}")

    before = profiling.snapshot(df)
    result = registry.execute(step["tool"], df, args)
    delta_text = ""
    status = result.status
    if result.mutated and status == "ok":
        after = profiling.snapshot(result.df)
        original_rows = len(state["original_df"])
        min_allowed_rows = original_rows * (1 - settings.max_row_drop_pct / 100.0)
        if after["rows"] < min_allowed_rows:
            status = "error"
            result.message = (
                f"guardrail: refused - would exceed the {settings.max_row_drop_pct:.0f}% "
                "row-deletion budget for the run"
            )
        else:
            delta = profiling.metrics_delta(before, after)
            delta_text = profiling.render_delta(delta)
            if delta:
                df = result.df
            else:
                status = "noop"

    step["status"] = "done" if status in ("ok", "noop") else "blocked"
    step["result"] = (delta_text or result.message)[:140]
    record = ActionRecord(
        seq=len(state["ledger"]) + 1, tool=step["tool"], args=args,
        status="ok" if status in ("ok", "noop") else "error",
        delta=delta_text,
        error=result.message if status == "error" else "",
        df_signature=profiling.df_signature(df),
    )
    return {
        "current_df": df,
        "cards": profiling.refresh_cards(df, state["cards"]),
        "plan_steps": steps,
        "current_step_index": index + 1,
        "ledger": [record.to_dict()],
        "logs": [record.render()],
        "tool_rounds": state["tool_rounds"] + 1,
    }


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
    return {"messages": [response], "tokens_used": usage_tokens(response)}


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
            # the row budget is cumulative for the whole run, so many small
            # drops cannot quietly delete half the dataset
            original_rows = len(state["original_df"])
            min_allowed_rows = original_rows * (1 - settings.max_row_drop_pct / 100.0)
            if after["rows"] < min_allowed_rows:
                total_dropped = original_rows - after["rows"]
                result = registry.ToolResult(
                    message=(
                        f"guardrail: refused - this call would bring the run's total deleted rows to "
                        f"{total_dropped} of {original_rows} ({100.0 * total_dropped / max(original_rows, 1):.0f}%), "
                        f"above the {settings.max_row_drop_pct:.0f}% budget for the whole run. "
                        "Repair values instead of dropping rows."
                    ),
                    status="error",
                )
            else:
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
            elif result.status == "noop":
                # the data is already in the state this step wanted - move on
                step["status"] = "done"
                step["result"] = "no change needed"
                step_index += 1
            elif result.status == "error":
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
