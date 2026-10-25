"""The main cleaning flow: upload -> context -> analysis -> plan review ->
live execution -> results. Sections reveal step by step."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui import charts
from app.ui.components import render_step_timeline, require_user, reset_workspace, run_store
from app.ui.theme import hero, section, stepper
from cleanai.core import profiling
from cleanai.services.pipeline import CleaningSession, revise_plan_with_chat, simulate_plan
from cleanai.services.recipe import build_plan_json, build_recipe_script
from cleanai.services.reporting import build_run_report

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

SAMPLES = {
    "HR employees (duplicates, fake nulls, wild ages)": "hr_employees.csv",
    "E-commerce orders (mixed dates, negative prices)": "ecommerce_orders.csv",
    "Survey responses (free-text chaos)": "survey_responses.csv",
}

CONTEXT_PRESETS = ["ML model training", "BI dashboard / reporting", "Audit & compliance", "General cleanup"]


def _stage() -> int:
    return st.session_state.get("ws_stage", 0)


def _set_stage(value: int) -> None:
    st.session_state.ws_stage = max(_stage(), value)


def workspace_page() -> None:
    user = require_user()
    st.markdown(stepper(_stage()), unsafe_allow_html=True)

    _render_upload()
    if "df" not in st.session_state:
        return
    _render_context()
    if not st.session_state.get("analysis_ready"):
        return
    if "summary" not in st.session_state:
        _render_review()
    else:
        _render_results(user)


# ---------------------------------------------------------------------------

def _load_dataframe(df: pd.DataFrame, name: str) -> None:
    reset_workspace()
    st.session_state.df = df
    st.session_state.file_name = name
    st.session_state.ws_stage = 1


def _render_upload() -> None:
    st.markdown(section(1, "Upload your data", "CSV in, clean data out. Start with a file or one of the messy samples."), unsafe_allow_html=True)
    upload_col, sample_col = st.columns([2.2, 1], gap="large")

    with upload_col:
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded is not None and st.session_state.get("file_name") != uploaded.name:
            try:
                _load_dataframe(pd.read_csv(uploaded), uploaded.name)
            except Exception as exc:
                st.error(f"Could not read that CSV: {exc}")

    with sample_col:
        st.caption("No file handy? Try a messy sample:")
        for label, filename in SAMPLES.items():
            if st.button(label, key=f"sample_{filename}", use_container_width=True):
                _load_dataframe(pd.read_csv(SAMPLES_DIR / filename), filename)
                st.rerun()

    if "df" in st.session_state:
        df = st.session_state.df
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", len(df.columns))
        missing_pct = 100.0 * df.isnull().sum().sum() / max(df.size, 1)
        m3.metric("Missing cells", f"{missing_pct:.1f}%")
        m4.metric("Duplicate rows", int(df.duplicated().sum()))
        with st.expander(f"Preview — {st.session_state.file_name}"):
            st.dataframe(df.head(50), use_container_width=True)


def _render_context() -> None:
    st.markdown(section(2, "What is this data for?", "The agent plans differently for ML, reporting, or audits."), unsafe_allow_html=True)
    preset = st.pills("Intended use", CONTEXT_PRESETS, key="context_preset")
    detail = st.text_input("Anything specific? (optional)",
                           placeholder="e.g. keep every row, salary must stay in rupees")
    context = " — ".join(x for x in [preset, detail] if x) or "general cleaning"
    st.session_state.context_text = context

    if not st.session_state.get("analysis_ready"):
        if st.button("Analyze dataset", type="primary"):
            _set_stage(2)
            session = CleaningSession(
                st.session_state.df.copy(), st.session_state.file_name, context,
            )
            with st.status("Agent is analyzing and planning…", expanded=True) as status:
                def on_event(node: str, output: dict) -> None:
                    for line in output.get("logs", []):
                        st.write(f"• {line}")
                session.run_until_review(on_event=on_event)
                status.update(label="Analysis and plan ready", state="complete")
            st.session_state.cleaning_session = session
            st.session_state.analysis_ready = True
            _set_stage(3)
            st.rerun()


def _render_review() -> None:
    session: CleaningSession = st.session_state.cleaning_session
    values = session.state_values()

    st.markdown(section(3, "What the agent found"), unsafe_allow_html=True)
    cards = values.get("cards", [])
    issue_rows = []
    for card in cards:
        for issue in card.issues_open:
            issue_rows.append({"Severity": issue.severity.upper(),
                               "Column": issue.column or "(table)",
                               "Issue": issue.kind.replace("_", " "),
                               "Evidence": issue.evidence})
    if issue_rows:
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No issues detected — this data already looks clean.")
    with st.expander("Full analysis report"):
        st.markdown(values.get("analysis_report", ""))
    if values.get("case_hints"):
        st.info(f"Memory from similar past runs:\n\n{values['case_hints']}")

    st.markdown(section(4, "Review the plan", "Edit any cell, untick steps you don't want, or tell the agent what to change in plain English."), unsafe_allow_html=True)

    plan = session.get_plan()
    editor_rows = [{
        "Run": step.get("status") != "skipped",
        "Step": step["id"],
        "Tool": step["tool"],
        "Columns": ", ".join(step.get("columns") or []),
        "Params": json.dumps(step.get("params") or {}),
        "Why": step.get("rationale", ""),
    } for step in plan]

    edited = st.data_editor(
        pd.DataFrame(editor_rows),
        column_config={
            "Run": st.column_config.CheckboxColumn(width="small"),
            "Step": st.column_config.TextColumn(disabled=True, width="small"),
            "Why": st.column_config.TextColumn(disabled=True),
        },
        hide_index=True, use_container_width=True, key="plan_editor",
    )

    with st.expander("Dry-run: what each step would change", expanded=True):
        impacts = st.session_state.get("plan_impacts")
        if impacts is None or st.button("Refresh impact preview"):
            impacts = simulate_plan(st.session_state.df, plan)
            st.session_state.plan_impacts = impacts
        for impact in impacts:
            icon = ":material/check_circle:" if impact["ok"] else ":material/warning:"
            st.markdown(f"{icon} **{impact['id']} {impact['tool']}** → {impact['effect']}")

    chat_instruction = st.chat_input("Tell the agent how to change the plan (e.g. 'don't drop any rows')")
    if chat_instruction:
        with st.spinner("Revising the plan…"):
            revised = revise_plan_with_chat(plan, chat_instruction, st.session_state.df)
        session.update_plan(revised)
        st.session_state.pop("plan_impacts", None)
        st.toast("Plan revised", icon=":material/edit:")
        st.rerun()

    run_col, discard_col = st.columns([1, 1])
    if run_col.button("Approve & run", type="primary", use_container_width=True):
        _apply_editor_changes(session, plan, edited)
        _execute(session)
    if discard_col.button("Discard run", use_container_width=True):
        reset_workspace()
        st.rerun()


def _apply_editor_changes(session: CleaningSession, plan: list[dict], edited: pd.DataFrame) -> None:
    by_id = {step["id"]: step for step in plan}
    final_steps = []
    for _, row in edited.iterrows():
        step = by_id.get(row["Step"])
        if step is None or not row["Run"]:
            continue
        step = dict(step)
        step["tool"] = str(row["Tool"]).strip()
        step["columns"] = [c.strip() for c in str(row["Columns"]).split(",") if c.strip()]
        try:
            step["params"] = json.loads(row["Params"] or "{}")
        except json.JSONDecodeError:
            pass
        final_steps.append(step)
    session.update_plan(final_steps)


def _execute(session: CleaningSession) -> None:
    _set_stage(4)
    timeline = st.empty()
    with st.status("Cleaning in progress…", expanded=True) as status:
        def on_event(node: str, output: dict) -> None:
            if "plan_steps" in output:
                render_step_timeline(output["plan_steps"], timeline)
            for line in output.get("logs", []):
                st.write(f"• {line}")
            if node == "validator":
                rounds = output.get("validation_rounds", "?")
                status.update(label=f"Validating… round {rounds}")
        session.resume(on_event=on_event)
        status.update(label="Run finished", state="complete")
    st.session_state.summary = session.result_summary()
    _set_stage(5)
    st.rerun()


def _render_results(user: dict) -> None:
    summary = st.session_state.summary
    session: CleaningSession = st.session_state.cleaning_session

    st.markdown(section(5, "Results"), unsafe_allow_html=True)

    lift = summary["score_after"] - summary["score_before"]
    notes = {
        "clean": "Every validation check passed. The dataset is ready for its job.",
        "partial": "The run finished with some remaining issues — the ledger below shows exactly what is left.",
        "stopped": summary["error"].replace("stopped:", "", 1).strip().capitalize() or "The run was stopped early.",
    }
    st.markdown(hero(
        summary["score_after"], lift, summary["status"],
        notes.get(summary["status"], "Run finished."),
    ), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing cells", summary["nulls_after"],
              delta=summary["nulls_after"] - summary["nulls_before"], delta_color="inverse")
    m2.metric("Duplicate rows", summary["duplicates_after"],
              delta=summary["duplicates_after"] - summary["duplicates_before"], delta_color="inverse")
    m3.metric("Rows kept", f"{summary['rows_after']:,}",
              delta=summary["rows_after"] - summary["rows_before"])
    m4.metric("LLM tokens", f"{summary['tokens_used']:,}")

    df_before = session.app.get_state(session.config).values["original_df"]
    df_after = session.state_values()["current_df"]

    chart_tab, scorecard_tab, diff_tab, ledger_tab = st.tabs([
        ":material/bar_chart: Charts",
        ":material/fact_check: Column scorecard",
        ":material/compare: Before / after",
        ":material/receipt_long: Action ledger",
    ])

    with chart_tab:
        c1, c2 = st.columns(2)
        nulls_fig = charts.nulls_before_after(df_before, df_after)
        if nulls_fig:
            c1.plotly_chart(nulls_fig, use_container_width=True)
        else:
            c1.info("No missing values in either version.")
        c2.plotly_chart(charts.quality_dimensions(
            profiling.dimension_scores(df_before), profiling.dimension_scores(df_after),
        ), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(charts.row_waterfall(
            summary["rows_before"], summary["ledger"], summary["rows_after"],
        ), use_container_width=True)
        c4.plotly_chart(charts.dtype_composition(df_before, df_after), use_container_width=True)

    with scorecard_tab:
        st.caption("Per-column health after the run. Every open issue and applied fix, column by column.")
        cards_after = session.state_values().get("cards", [])
        nulls_before = df_before.isnull().sum()
        rows = []
        for card in cards_after:
            before_nulls = int(nulls_before.get(card.name, 0))
            rows.append({
                "Column": card.name,
                "Role": card.role,
                "Status": card.status,
                "Nulls before": before_nulls,
                "Nulls after": card.null_count,
                "Open issues": "; ".join(i.kind.replace("_", " ") for i in card.issues_open) or "—",
                "Fixes applied": ", ".join(card.fixes_applied[-4:]) or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with diff_tab:
        st.caption("First 40 rows side by side. Changed cells in the after view are highlighted.")
        left, right = st.columns(2)
        sample_before = df_before.head(40).reset_index(drop=True)
        sample_after = df_after.head(40).reset_index(drop=True)
        left.markdown("**Before**")
        left.dataframe(sample_before, use_container_width=True)
        right.markdown("**After**")
        shared_cols = [c for c in sample_after.columns if c in sample_before.columns]
        overlap = min(len(sample_before), len(sample_after))
        if shared_cols and overlap > 0:
            before_text = sample_before.loc[:overlap - 1, shared_cols].astype(str)

            def _highlight(frame: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
                for col in shared_cols:
                    changed = frame[col].astype(str).iloc[:overlap].ne(before_text[col])
                    styles.loc[changed[changed].index, col] = "background-color: #E5E4FB"
                return styles

            right.dataframe(sample_after.style.apply(_highlight, axis=None), use_container_width=True)
        else:
            right.dataframe(sample_after, use_container_width=True)

    with ledger_tab:
        for record in summary["ledger"]:
            icon = {"ok": ":material/check_circle:", "error": ":material/error:", "noop": ":material/remove:"}.get(record["status"], ":material/circle:")
            st.markdown(f"{icon} **#{record['seq']} {record['tool']}** "
                        f"`{json.dumps(record['args'], default=str)[:100]}`")
            if record.get("delta"):
                st.caption(f"effect: {record['delta']}")
            if record.get("error"):
                st.caption(f"error: {record['error']}")

    st.markdown(section(6, "Take it with you", "Everything is reusable: the data, the report, and a script that re-runs this cleaning anywhere."), unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Cleaned CSV", summary["cleaned_csv"], icon=":material/download:",
                       file_name=f"cleaned_{summary['dataset_name']}", type="primary",
                       use_container_width=True)
    d2.download_button("Recipe script (.py)", build_recipe_script(summary), icon=":material/code:",
                       file_name="clean_recipe.py", use_container_width=True)
    d3.download_button("Plan (.json)", build_plan_json(summary), icon=":material/data_object:",
                       file_name="cleaning_plan.json", use_container_width=True)
    d4.download_button("Run report (.md)", build_run_report(summary), icon=":material/description:",
                       file_name="run_report.md", use_container_width=True)

    if "saved_run_id" not in st.session_state:
        st.session_state.saved_run_id = run_store().save_run(user["id"], summary)
        try:
            session.learn_from_run()
        except Exception:
            pass
        st.toast("Run saved to history", icon=":material/save:")

    if st.button("Start a new run"):
        reset_workspace()
        st.rerun()
