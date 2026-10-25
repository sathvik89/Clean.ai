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
        ["📋 Plan", "🧾 Actions", "🔍 Analysis", "🪵 Logs"])

    with plan_tab:
        for step in run["steps"]:
            icon = {"done": "✅", "blocked": "🚧", "skipped": "⏭️"}.get(step.get("status"), "⚪")
            st.markdown(f"{icon} **{step['id']} · {step['tool']}** on {step.get('columns') or 'table'} "
                        f"— {step.get('rationale', '')}")
            if step.get("result"):
                st.caption(step["result"])

    with ledger_tab:
        for record in run["ledger"]:
            icon = {"ok": "✅", "error": "❌", "noop": "➖"}.get(record["status"], "•")
            st.markdown(f"{icon} **#{record['seq']} {record['tool']}** "
                        f"`{json.dumps(record.get('args', {}), default=str)[:100]}`")
            if record.get("delta"):
                st.caption(f"effect: {record['delta']}")

    with report_tab:
        st.markdown(run.get("analysis_report") or "_no analysis stored_")

    with logs_tab:
        st.code(run.get("logs_text") or "no logs", language=None)

    st.divider()
    d1, d2, d3 = st.columns(3)
    summary = {**run, "dataset_name": run["dataset_name"]}
    d1.download_button("⬇️ Cleaned CSV", run.get("cleaned_csv") or "",
                       file_name=f"cleaned_{run['dataset_name']}", type="primary",
                       use_container_width=True)
    d2.download_button("🧾 Recipe script", build_recipe_script(summary),
                       file_name="clean_recipe.py", use_container_width=True)
    d3.download_button("📝 Run report", build_run_report(summary),
                       file_name="run_report.md", use_container_width=True)
