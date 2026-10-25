"""EDA charts rendered straight from the deterministic EDA bundle.

The agent narrates; these charts prove it. Everything draws from
pre-computed numbers, so rendering is instant even for huge files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui.charts import GRID, LAYOUT_DEFAULTS
from app.ui.theme import ACCENT, PRIMARY, SECONDARY, TEXT_MUTED


def missingness(bundle: Dict[str, Any]) -> Optional[go.Figure]:
    rows = bundle.get("missing", [])[:12]
    if not rows:
        return None
    rows = sorted(rows, key=lambda m: m["pct"])
    fig = go.Figure(go.Bar(
        y=[m["column"] for m in rows],
        x=[m["pct"] for m in rows],
        orientation="h",
        marker=dict(color=PRIMARY),
        text=[f"{m['pct']}%" for m in rows],
        textposition="outside",
        textfont=dict(size=11.5, color=TEXT_MUTED),
        hovertemplate="%{y}: %{customdata:,} missing (%{x}%)<extra></extra>",
        customdata=[m["count"] for m in rows],
    ))
    fig.update_layout(title="Missing values by column", showlegend=False, **LAYOUT_DEFAULTS)
    fig.update_xaxes(title="% of rows missing", range=[0, max(m["pct"] for m in rows) * 1.25], **GRID)
    fig.update_yaxes(**GRID)
    return fig


def distributions(bundle: Dict[str, Any]) -> Optional[go.Figure]:
    numeric = bundle.get("numeric", [])[:4]
    if not numeric:
        return None
    cols = 2 if len(numeric) > 1 else 1
    rows = (len(numeric) + 1) // 2
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[n["column"] for n in numeric],
                        vertical_spacing=0.18)
    for index, item in enumerate(numeric):
        edges, counts = item["hist"]["edges"], item["hist"]["counts"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        fig.add_trace(
            go.Bar(x=centers, y=counts, marker=dict(color=PRIMARY), showlegend=False,
                   hovertemplate="%{x:.4g}: %{y:,} rows<extra></extra>"),
            row=index // cols + 1, col=index % cols + 1,
        )
    fig.update_layout(title="Numeric distributions", bargap=0.08, **LAYOUT_DEFAULTS)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    fig.update_annotations(font=dict(size=12.5, color=TEXT_MUTED))
    return fig


def correlation_heatmap(bundle: Dict[str, Any]) -> Optional[go.Figure]:
    matrix = bundle.get("corr_matrix")
    if not matrix:
        return None
    fig = go.Figure(go.Heatmap(
        z=matrix["values"], x=matrix["columns"], y=matrix["columns"],
        zmin=-1, zmax=1,
        colorscale=[[0.0, ACCENT], [0.5, "#F1F3F9"], [1.0, PRIMARY]],
        colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(size=11, color=TEXT_MUTED)),
        text=[[f"{v:.2f}" for v in row] for row in matrix["values"]],
        texttemplate="%{text}",
        textfont=dict(size=10.5),
        hovertemplate="%{y} ~ %{x}: r=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Correlations between numeric columns", **LAYOUT_DEFAULTS)
    fig.update_xaxes(tickfont=dict(size=11, color=TEXT_MUTED))
    fig.update_yaxes(tickfont=dict(size=11, color=TEXT_MUTED), autorange="reversed")
    return fig


def category_bars(bundle: Dict[str, Any]) -> Optional[go.Figure]:
    cats = bundle.get("categorical", [])[:2]
    if not cats:
        return None
    fig = make_subplots(rows=1, cols=len(cats),
                        subplot_titles=[f"{c['column']} ({c['n_unique']} unique)" for c in cats])
    palette = [PRIMARY, SECONDARY]
    for index, item in enumerate(cats):
        top = list(reversed(item["top"]))
        fig.add_trace(
            go.Bar(y=[t["value"] for t in top], x=[t["count"] for t in top],
                   orientation="h", marker=dict(color=palette[index % 2]), showlegend=False,
                   hovertemplate="%{y}: %{x:,} rows<extra></extra>"),
            row=1, col=index + 1,
        )
    fig.update_layout(title="Most frequent category values", **LAYOUT_DEFAULTS)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(tickfont=dict(size=11.5, color=TEXT_MUTED))
    fig.update_annotations(font=dict(size=12.5, color=TEXT_MUTED))
    return fig
