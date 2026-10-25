"""Deterministic exploratory data analysis.

Computes everything the EDA charts and the agent's investigation need,
entirely in pandas. Large frames are sampled for the expensive parts so
a million-row file explores as fast as a hundred-row one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cleanai.core.profiling import ColumnCard, build_cards, mask_value

SAMPLE_THRESHOLD = 60_000
SAMPLE_SIZE = 50_000
MAX_HIST_COLUMNS = 6
MAX_CORR_COLUMNS = 12


def _numeric_view(df: pd.DataFrame, cards: List[ColumnCard]) -> pd.DataFrame:
    """Numeric columns plus numeric-roled text columns, coerced."""
    out = {}
    for card in cards:
        if card.role != "numeric":
            continue
        series = pd.to_numeric(df[card.name], errors="coerce")
        if series.notna().sum() > 0:
            out[card.name] = series
    return pd.DataFrame(out)


def compute_bundle(df: pd.DataFrame, cards: Optional[List[ColumnCard]] = None) -> Dict[str, Any]:
    cards = cards or build_cards(df)
    total_rows = len(df)
    sampled = total_rows > SAMPLE_THRESHOLD
    sample = df.sample(SAMPLE_SIZE, random_state=7) if sampled else df

    bundle: Dict[str, Any] = {
        "rows": total_rows,
        "cols": len(df.columns),
        "sampled": sampled,
        "sample_rows": len(sample),
        "duplicates": int(df.duplicated().sum()),
        "missing": [],
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "correlations": [],
        "corr_matrix": None,
    }

    # missingness on the full frame - it's cheap and must be exact
    null_counts = df.isnull().sum()
    for col in df.columns:
        count = int(null_counts[col])
        if count > 0:
            bundle["missing"].append({
                "column": str(col),
                "count": count,
                "pct": round(100.0 * count / max(total_rows, 1), 2),
            })
    bundle["missing"].sort(key=lambda m: -m["count"])

    numeric = _numeric_view(sample, cards)

    for col in list(numeric.columns)[:MAX_HIST_COLUMNS * 2]:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        counts, edges = np.histogram(series, bins=min(24, max(8, int(np.sqrt(len(series))))))
        bundle["numeric"].append({
            "column": str(col),
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 3),
            "q1": round(float(q1), 3),
            "median": round(float(series.median()), 3),
            "q3": round(float(q3), 3),
            "max": round(float(series.max()), 3),
            "skew": round(float(series.skew()), 3) if len(series) > 2 else 0.0,
            "outliers": outliers,
            "hist": {
                "edges": [round(float(e), 4) for e in edges],
                "counts": [int(c) for c in counts],
            },
        })

    for card in cards:
        if card.role not in ("categorical", "text") or card.n_unique > 60:
            continue
        counts = sample[card.name].dropna().astype(str).value_counts().head(6)
        if counts.empty:
            continue
        bundle["categorical"].append({
            "column": card.name,
            "n_unique": card.n_unique,
            "top": [{"value": mask_value(str(v))[:34], "count": int(c)} for v, c in counts.items()],
        })

    for card in cards:
        if card.role != "datetime":
            continue
        parsed = pd.to_datetime(sample[card.name].astype(str), errors="coerce", format="mixed")
        valid = parsed.dropna()
        if valid.empty:
            continue
        bundle["datetime"].append({
            "column": card.name,
            "min": str(valid.min())[:19],
            "max": str(valid.max())[:19],
            "invalid": int(parsed.isna().sum() - sample[card.name].isna().sum()),
        })

    if len(numeric.columns) >= 2:
        corr_cols = list(numeric.columns)[:MAX_CORR_COLUMNS]
        corr = numeric[corr_cols].corr()
        bundle["corr_matrix"] = {
            "columns": [str(c) for c in corr_cols],
            "values": [[round(float(v), 3) if pd.notna(v) else 0.0 for v in row] for row in corr.values],
        }
        pairs = []
        for i, a in enumerate(corr_cols):
            for j in range(i + 1, len(corr_cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.3:
                    pairs.append({"a": str(a), "b": str(corr_cols[j]), "r": round(float(r), 3)})
        pairs.sort(key=lambda p: -abs(p["r"]))
        bundle["correlations"] = pairs[:10]

    return bundle


def render_for_llm(bundle: Dict[str, Any]) -> str:
    """Compact, token-friendly EDA summary for the agent."""
    lines = [f"shape: {bundle['rows']} rows x {bundle['cols']} cols"
             + (f" (statistics computed on a {bundle['sample_rows']}-row sample)" if bundle["sampled"] else "")]
    lines.append(f"exact duplicate rows: {bundle['duplicates']}")

    if bundle["missing"]:
        top = ", ".join(f"{m['column']}={m['count']} ({m['pct']}%)" for m in bundle["missing"][:10])
        lines.append(f"missing: {top}")
    else:
        lines.append("missing: none")

    for item in bundle["numeric"][:MAX_HIST_COLUMNS * 2]:
        lines.append(
            f"num {item['column']}: mean={item['mean']} median={item['median']} "
            f"min={item['min']} max={item['max']} skew={item['skew']} outliers={item['outliers']}"
        )
    for item in bundle["categorical"][:8]:
        top = ", ".join(f"{t['value']}({t['count']})" for t in item["top"][:4])
        lines.append(f"cat {item['column']}: {item['n_unique']} unique | top: {top}")
    for item in bundle["datetime"]:
        lines.append(f"date {item['column']}: {item['min']} .. {item['max']}, unparseable={item['invalid']}")
    if bundle["correlations"]:
        pairs = ", ".join(f"{p['a']}~{p['b']}:{p['r']}" for p in bundle["correlations"][:6])
        lines.append(f"strongest correlations: {pairs}")
    return "\n".join(lines)
