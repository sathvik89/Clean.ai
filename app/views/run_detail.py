import json

import streamlit as st

from app.ui.components import require_user, run_store
from cleanai.services.recipe import build_recipe_script
from cleanai.services.reporting import build_run_report


def run_detail_page() -> None:
    user = require_user()
    run_id = st.session_state.get("selected_run_id")
    if not run_id:
        st.info("Pick a run from the History page to see its details.")
        return

    run = run_store().get_run(run_id, user["id"])
    if not run:
        st.error("That run could not be found.")
        return

    st.title(f"Run #{run['id']} — {run['dataset_name']}")
    st.caption(f"{run['created_at'][:16].replace('T', ' ')} · goal: {run['user_context'] or 'general cleaning'}")

    lift = (run["score_after"] or 0) - (run["score_before"] or 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quality score", run["score_after"], delta=f"{lift:+d}")
    m2.metric("Missing cells", run["nulls_after"],
              delta=(run["nulls_after"] or 0) - (run["nulls_before"] or 0), delta_color="inverse")
    m3.metric("Duplicates", run["duplicates_after"],
              delta=(run["duplicates_after"] or 0) - (run["duplicates_before"] or 0), delta_color="inverse")
    m4.metric("Rows", run["rows_after"], delta=(run["rows_after"] or 0) - (run["rows_before"] or 0))
    if run.get("error"):
        st.warning(f"Stopped early: {run['error']}")

    plan_tab, ledger_tab, report_tab, logs_tab = st.tabs(
        [":material/checklist: Plan", ":material/receipt_long: Actions", ":material/search: Analysis", ":material/terminal: Logs"])

    with plan_tab:
        for step in run["steps"]:
            icon = {"done": ":material/check_circle:", "blocked": ":material/block:", "skipped": ":material/skip_next:"}.get(step.get("status"), ":material/radio_button_unchecked:")
            st.markdown(f"{icon} **{step['id']} · {step['tool']}** on {step.get('columns') or 'table'} "
                        f"— {step.get('rationale', '')}")
            if step.get("result"):
                st.caption(step["result"])

    with ledger_tab:
        for record in run["ledger"]:
            icon = {"ok": ":material/check_circle:", "error": ":material/error:", "noop": ":material/remove:"}.get(record["status"], ":material/circle:")
            st.markdown(f"{icon} **#{record['seq']} {record['tool']}** "
                        f"`{json.dumps(record.get('args', {}), default=str)[:100]}`")
            if record.get("delta"):
                st.caption(f"effect: {record['delta']}")

    with report_tab:
        st.markdown(run.get("analysis_report") or "_no analysis stored_")

    with logs_tab:
        st.code(run.get("logs_text") or "no logs", language=None)

    st.divider()
    st.markdown("#### Re-apply this recipe to a new file")
    st.caption("Replays this run's effective actions on a fresh CSV — deterministic, no LLM, no waiting.")
    new_file = st.file_uploader("New CSV", type=["csv"], key=f"replay_{run['id']}",
                                label_visibility="collapsed")
    if new_file is not None:
        import pandas as pd

        from cleanai.services.pipeline import apply_recorded_actions

        try:
            new_df = pd.read_csv(new_file)
            replay = apply_recorded_actions(new_df, run["ledger"])
        except Exception as exc:
            st.error(f"Could not apply the recipe: {exc}")
        else:
            r1, r2, r3 = st.columns([1, 1, 2])
            r1.metric("Quality", replay["score_after"],
                      delta=replay["score_after"] - replay["score_before"])
            r2.metric("Rows", len(replay["df"]), delta=len(replay["df"]) - len(new_df))
            r3.download_button("Download cleaned file", replay["df"].to_csv(index=False),
                               file_name=f"cleaned_{new_file.name}", type="primary",
                               icon=":material/download:", use_container_width=True)
            with st.expander("Replay log"):
                for line in replay["log"]:
                    st.markdown(f"- {line}")

    st.divider()
    d1, d2, d3 = st.columns(3)
    summary = {**run, "dataset_name": run["dataset_name"]}
    d1.download_button("Cleaned CSV", run.get("cleaned_csv") or "", icon=":material/download:",
                       file_name=f"cleaned_{run['dataset_name']}", type="primary",
                       use_container_width=True)
    d2.download_button("Recipe script", build_recipe_script(summary), icon=":material/code:",
                       file_name="clean_recipe.py", use_container_width=True)
    d3.download_button("Run report", build_run_report(summary), icon=":material/description:",
                       file_name="run_report.md", use_container_width=True)
