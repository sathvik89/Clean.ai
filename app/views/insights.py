from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.components import require_user, run_store
from app.ui.charts import GRID, LAYOUT_DEFAULTS
from app.ui.theme import PRIMARY, SECONDARY


def insights_page() -> None:
    user = require_user()
    st.title("Insights")

    runs = run_store().list_runs(user["id"], limit=500)
    if not runs:
        st.info("Insights appear once you have completed a few runs.")
        return

    lifts = [(r["score_after"] or 0) - (r["score_before"] or 0) for r in runs]
    nulls_fixed = sum((r["nulls_before"] or 0) - (r["nulls_after"] or 0) for r in runs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total runs", len(runs))
    m2.metric("Clean finishes", sum(1 for r in runs if r["status"] == "clean"))
    m3.metric("Avg quality lift", f"+{sum(lifts) / len(lifts):.0f}")
    m4.metric("Missing cells fixed", f"{nulls_fixed:,}")

    frame = pd.DataFrame(runs)
    frame["day"] = frame["created_at"].str[:10]
    by_day = frame.groupby("day").size().reset_index(name="runs")

    fig = go.Figure(go.Bar(x=by_day["day"], y=by_day["runs"], marker=dict(color=PRIMARY),
                           hovertemplate="%{x}: %{y} runs<extra></extra>"))
    fig.update_layout(title="Runs per day", showlegend=False, **LAYOUT_DEFAULTS)
    fig.update_xaxes(type="category", **GRID)
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, use_container_width=True)

    usage = run_store().tool_usage(user["id"])
    if usage:
        fig_tools = go.Figure(go.Bar(
            y=[t.replace("_", " ") for t in usage.keys()], x=list(usage.values()),
            orientation="h", marker=dict(color=PRIMARY),
            hovertemplate="%{y}: %{x} effective uses<extra></extra>",
        ))
        fig_tools.update_layout(title="Which tools actually fix your data", showlegend=False, **LAYOUT_DEFAULTS)
        fig_tools.update_xaxes(title="times a call changed the data", **GRID)
        fig_tools.update_yaxes(**GRID)
        st.plotly_chart(fig_tools, use_container_width=True)

    scores = frame.dropna(subset=["score_before", "score_after"])
    fig2 = go.Figure()
    fig2.add_scatter(x=list(range(1, len(scores) + 1)), y=scores["score_before"][::-1],
                     mode="lines+markers", name="Before", line=dict(color="#94A3B8", width=2))
    fig2.add_scatter(x=list(range(1, len(scores) + 1)), y=scores["score_after"][::-1],
                     mode="lines+markers", name="After", line=dict(color=SECONDARY, width=2))
    fig2.update_layout(title="Quality scores per run", **LAYOUT_DEFAULTS)
    fig2.update_xaxes(title="run", **GRID)
    fig2.update_yaxes(range=[0, 105], **GRID)
    st.plotly_chart(fig2, use_container_width=True)
