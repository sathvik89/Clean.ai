"""Plotly chart builders for before/after data quality.

Every chart keeps one axis, a white surface, recessive gridlines, and the
fixed entity colors: before = neutral gray, after = indigo.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from app.ui.theme import ACCENT, AFTER_COLOR, BEFORE_COLOR, BORDER, PRIMARY, SURFACE, TEXT, TEXT_MUTED

LAYOUT_DEFAULTS = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(color=TEXT, size=13),
    margin=dict(l=10, r=10, t=76, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
    title=dict(yanchor="top", y=0.97),
    hovermode="closest",
)

GRID = dict(gridcolor=BORDER, zerolinecolor=BORDER)


def nulls_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame) -> go.Figure:
    """Horizontal grouped bars of null counts per column, sorted by severity."""
    nulls_before = df_before.isnull().sum()
    shared_columns = [c for c in df_before.columns if c in df_after.columns]
    rows = [(c, int(nulls_before[c]), int(df_after[c].isnull().sum())) for c in shared_columns]
    rows = [r for r in rows if r[1] > 0 or r[2] > 0]
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]

    fig = go.Figure()
    fig.add_bar(y=names, x=[r[1] for r in rows], name="Before", orientation="h",
                marker=dict(color=BEFORE_COLOR), hovertemplate="%{y}: %{x} nulls before<extra></extra>")
    fig.add_bar(y=names, x=[r[2] for r in rows], name="After", orientation="h",
                marker=dict(color=AFTER_COLOR), hovertemplate="%{y}: %{x} nulls after<extra></extra>")
    fig.update_layout(title="Missing values per column", barmode="group",
                      bargap=0.3, **LAYOUT_DEFAULTS)
    fig.update_xaxes(title="null count", **GRID)
    fig.update_yaxes(**GRID)
    return fig


def quality_dimensions(before: Dict[str, int], after: Dict[str, int]) -> go.Figure:
    labels = [k.capitalize() for k in before]
    fig = go.Figure()
    fig.add_bar(x=labels, y=list(before.values()), name="Before",
                marker=dict(color=BEFORE_COLOR), hovertemplate="%{x}: %{y}/100 before<extra></extra>")
    fig.add_bar(x=labels, y=list(after.values()), name="After",
                marker=dict(color=AFTER_COLOR),
                text=list(after.values()), textposition="outside",
                hovertemplate="%{x}: %{y}/100 after<extra></extra>")
    fig.update_layout(title="Quality dimensions (0-100)", barmode="group",
                      bargap=0.35, **LAYOUT_DEFAULTS)
    fig.update_yaxes(range=[0, 112], **GRID)
    fig.update_xaxes(**GRID)
    return fig


def row_waterfall(rows_before: int, ledger: List[dict], rows_after: int) -> go.Figure:
    """Where did the rows go: start -> per-action drops -> final."""
    labels, values, measures = ["Input rows"], [rows_before], ["absolute"]
    for record in ledger:
        delta = record.get("delta", "")
        if "rows " in delta and record.get("status") == "ok":
            fragment = delta.split("rows ")[1].split(",")[0]
            try:
                before_n, after_n = (int(x) for x in fragment.split("->"))
            except ValueError:
                continue
            if after_n != before_n:
                labels.append(record["tool"].replace("_", " "))
                values.append(after_n - before_n)
                measures.append("relative")
    labels.append("Final rows")
    values.append(rows_after)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        decreasing=dict(marker=dict(color=ACCENT)),
        increasing=dict(marker=dict(color=PRIMARY)),
        totals=dict(marker=dict(color=PRIMARY)),
        connector=dict(line=dict(color=BORDER)),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Row count through the run", showlegend=False, **LAYOUT_DEFAULTS)
    fig.update_yaxes(**GRID)
    fig.update_xaxes(**GRID)
    return fig


def dtype_composition(df_before: pd.DataFrame, df_after: pd.DataFrame) -> go.Figure:
    """Two stacked horizontal bars showing how column types changed."""
    def counts(df: pd.DataFrame) -> Dict[str, int]:
        buckets: Dict[str, int] = {}
        for dtype in df.dtypes:
            name = str(dtype)
            bucket = ("numeric" if any(t in name.lower() for t in ("int", "float")) else
                      "datetime" if "datetime" in name else
                      "text" if name in ("object", "str", "string") or "string" in name.lower() else
                      "category" if "category" in name else name)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        return buckets

    before_counts, after_counts = counts(df_before), counts(df_after)
    kinds = sorted(set(before_counts) | set(after_counts))
    from app.ui.theme import CATEGORICAL
    palette = {kind: CATEGORICAL[i % len(CATEGORICAL)] for i, kind in enumerate(kinds)}

    fig = go.Figure()
    for kind in kinds:
        fig.add_bar(
            y=["After", "Before"],
            x=[after_counts.get(kind, 0), before_counts.get(kind, 0)],
            name=kind, orientation="h",
            marker=dict(color=palette[kind], line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{kind}: %{{x}} columns<extra></extra>",
        )
    fig.update_layout(title="Column types, before vs after", barmode="stack", **LAYOUT_DEFAULTS)
    fig.update_xaxes(title="columns", **GRID)
    fig.update_yaxes(**GRID)
    return fig
