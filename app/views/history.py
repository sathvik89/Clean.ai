import pandas as pd
import streamlit as st

from app.ui.components import require_user, run_store


def history_page() -> None:
    user = require_user()
    st.title("Run history")

    runs = run_store().list_runs(user["id"])
    if not runs:
        st.info("No runs yet — clean your first dataset from the Workspace.")
        if st.button("Go to Workspace", type="primary"):
            st.session_state.nav_hint = "workspace"
            st.rerun()
        return

    total_lift = sum((r["score_after"] or 0) - (r["score_before"] or 0) for r in runs)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total runs", len(runs))
    m2.metric("Avg quality lift", f"+{total_lift / len(runs):.0f}")
    m3.metric("Rows processed", f"{sum(r['rows_before'] or 0 for r in runs):,}")

    search = st.text_input("Filter by dataset name", placeholder="orders.csv")
    visible = [r for r in runs if search.lower() in r["dataset_name"].lower()] if search else runs

    table = pd.DataFrame([{
        "ID": r["id"],
        "Dataset": r["dataset_name"],
        "When": r["created_at"][:16].replace("T", " "),
        "Status": r["status"],
        "Rows in→out": f"{r['rows_before']}→{r['rows_after']}",
        "Quality before": r["score_before"],
        "Quality after": r["score_after"],
        "Steps": r["n_steps"],
    } for r in visible])

    selection = st.dataframe(
        table, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Quality before": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "Quality after": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        },
    )
    rows_selected = selection.selection.rows if selection and selection.selection else []
    if rows_selected:
        st.session_state.selected_run_id = int(table.iloc[rows_selected[0]]["ID"])
        st.switch_page(st.session_state.run_detail_page)
